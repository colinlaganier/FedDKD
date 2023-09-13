import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from knowledge_distillation import Logits, SoftTarget
import numpy as np

class Client:

    def __init__(self, client_id, device, model, dataloader, dataset_id, params, checkpoint_path, logger):
        # Client properties
        self.id = client_id
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.round = -1 # ignore initial round

        self.dataloader = dataloader
        self.dataset_id = dataset_id

        self.model = model
        self.criterion = None
        self.optimizer = None
        self.diffusion_model = None
        self.diffusion_seed = None  
        self.device = device
        self.eval_seed = None
        self.synthetic_dataset = None
        self.epochs = None
        self.params = params
        self.kd_alpha = None

        # Knowledge distillation strategy
        self.strategies = {
            0: self.constant,
            1: self.linear,
            2: self.cumulative_exponential,
            3: self.exponential,
            4: lambda x: self.sigmoid(x, 25),
            5: lambda x: self.sigmoid(x, 15)
        }
        if self.params["kd_scheduling"] is not None:
            self.kd_scheduling = self.strategies[self.params["kd_scheduling"]]
        else:
            self.kd_scheduling = None

    def get_alpha(self):
        return round(self.kd_scheduling(self.round), 2)

    def constant(self, x):
        return 0.5

    def linear(self, x):
        return 0.1 + 0.8 * x / 50

    def cumulative_exponential(self, x):
        return 1 - (0.1 + 0.8 * np.exp(-x/10))

    def exponential(self, x):
        return 0.1 + 0.1 * np.exp(0.1 * x) / 18.85

    def sigmoid(self, x, shift):
        return 0.8 / (1 + np.exp(-0.2 * (x - shift)))

    def init_client(self, load_checkpoint=False):
        """
        Initialize the client model training process

        Args:
            load_checkpoint (bool): whether to load a pretrained checkpoint
        """
        print(f"Initializing client {self.id}")
        
        torch.manual_seed(self.id)
        num_channels = 3 if self.dataset_id == "cinic10" else 1
        self.model = self.model(num_channels, self.params["num_classes"])
        if torch.cuda.device_count() > 1:
            print("Using multiple GPUs")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.optimizer = self.params["optimizer"](self.model.parameters(),
                                             lr=self.params["lr"], 
                                             momentum=self.params["momentum"],
                                            #  weight_decay=self.params["weight_decay"]
                                             )
        self.criterion = self.params["criterion"]().to(self.device)
        if load_checkpoint:
            self.load_checkpoint(f"checkpoints/pretrain/cinic10/iid/resnet/client_0{self.id}.pt")
        else:
            print(self.params["pretrain_epochs"])
            self.train(epochs=self.params["pretrain_epochs"])

    def train(self, epochs=None):
        """
        Train the client model

        Args:
            epochs (int): number of epochs to train for
        """
        torch.manual_seed(self.id)
        self.model.train()
        self.round += 1

        num_epochs = epochs if epochs else self.params["epochs"]

        for epoch in range(num_epochs):
            # Set statistic variables
            total_loss = 0
            total_correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(self.dataloader):
                # Send data and target to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero out gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()

                total += target.size(0)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                    
            # past_epochs = epoch + epochs if self.round == 0 else epoch
            # Log statistics
            self.logger.add_scalar(f"Training_Loss/Client_{self.id:02}", total_loss/len(self.dataloader), (self.round - 1) * self.params["epochs"] + epoch + self.params["pretrain_epochs"])
            self.logger.add_scalar(f"Training_Accuracy/Client_{self.id:02}", 100*total_correct/total, (self.round - 1) * self.params["epochs"] + epoch + self.params["pretrain_epochs"])

        
        self.logger.flush()

    def knowledge_distillation(self, server_logits, synthetic_data=None, diffusion_seed=None):
        """
        Knowledge distillation from server to client

        Args:
            server_logits (torch.Tensor): logits from the server model
            synthetic_data (torch.utils.data.DataLoader): synthetic diffusion data - if not generated at runtime
            diffusion_seed (int): random seed for diffusion sampling - if generated at runtime
        """

        # Generate synthetic data if not provided
        if synthetic_data is None:
            synthetic_data = self.synthetic_dataset
        
        # Set client seed for consistency
        torch.manual_seed(self.id)

        self.model.train()

        kd_criterion = self.params["kd_criterion"](self.params["kd_temperature"]).to(self.device)
        criterion = self.params["criterion"]().to(self.device)

        # Set statistic variables
        kd_total_loss = 0
        total_loss = 0

        if self.kd_scheduling is not None: 
            alpha = self.get_alpha()
        else:
            alpha = self.params["kd_alpha"]

        for epoch in range(self.params["kd_epochs"]):
            kd_total_loss = 0
            cls_total_loss = 0

            for batch_idx, ((data, target), logit) in enumerate(zip(synthetic_data, server_logits)):
                logit = logit[0]
                data, target, logit = data.to(self.device), target.to(self.device), logit.to(self.device)
                
                self.optimizer.zero_grad()

                output = self.model(data)

                kd_loss = kd_criterion(output, logit)
                cls_loss = criterion(output, target)
                
                # Knowledge distillation alpha scheduling
                if self.kd_scheduling is not None: 
                    alpha = self.get_alpha()
                    loss = (1 - alpha) * cls_loss + alpha * kd_loss
                else:
                    loss = (1 - self.params["kd_alpha"]) * cls_loss + self.params["kd_alpha"] * kd_loss

                kd_total_loss += kd_loss.item()
                cls_total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            # Log statistics
            self.logger.add_scalar(f"KD_Loss/Client_{self.id:02}", kd_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)
            self.logger.add_scalar(f"KD_Class_Loss/Client_{self.id:02}", cls_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)

        self.logger.flush()
        # del optimizer, kd_criterion, criterion

    def generate_logits(self, synthetic_data=None, diffusion_seed=None):
        """
        Generate logits from the client model

        Args:
            synthetic_data (torch.utils.data.DataLoader): synthetic diffusion data - if not generated at runtime
            diffusion_seed (int): random seed for diffusion sampling - if generated at runtime
        Returns:
            torch.Tensor: logits from the client model
        """
        torch.manual_seed(self.id)

        if synthetic_data is None:
            synthetic_data = self.synthetic_dataset

        # self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = []

            for batch_idx, (data, target) in enumerate(synthetic_data):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                logits.append(output)

        # return torch.cat(logits, dim=0) 
        return torch.cat(logits).detach().cpu()


    @torch.inference_mode()
    def evaluate(self, test_dataloader, post_kd=False):
        """
        Evaluate the client model on the test dataset

        Args:
            test_dataloader (torch.utils.data.DataLoader): test dataset
            post_kd (bool): whether evaluation of model is after knowledge distillation
        """
        torch.manual_seed(self.params["eval_seed"])
        self.model.eval()
        
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total += target.size(0)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()

         # Log statistics
            self.logger.add_scalar(f"Validation_Loss/Client_{self.id:02}", total_loss/len(test_dataloader), self.round)
            self.logger.add_scalar(f"Validation_Accuracy/Client_{self.id:02}", 100*total_correct/total, self.round)
            self.logger.flush()

    def save_checkpoint(self, checkpoint_path = None):
        """
        Save the client model checkpoint
        """
        path = checkpoint_path if checkpoint_path is not None else self.checkpoint_path

        torch.save({
            'round': self.round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path + f"/client_{self.id:02}.pt")
        
    def load_checkpoint(self, checkpoint):
        """
        Load the client model checkpoint

        Args:
            checkpoint (str): path to the checkpoint
        """
        checkpoint = torch.load(checkpoint)
        self.round = checkpoint['round']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def set_synthetic_dataset(self, synthetic_dataset):
        """
        Set the synthetic dataset

        Args:
            synthetic_dataset (torch.utils.data.DataLoader): synthetic diffusion data
        """
        self.synthetic_dataset = synthetic_dataset