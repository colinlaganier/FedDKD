import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from diffusion import DiT
from knowledge_distillation import Logits, SoftTarget
import numpy as np

class Client:

    def __init__(self, client_id, device, model, dataloader, params, checkpoint_path, logger):
        # Client properties
        self.id = client_id
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.round = -1 # ignore initial round

        self.dataloader = dataloader
        self.model = model
        # self.model = deepcopy(model).to(self.gpu_id)
        self.criterion = None
        self.optimizer = None
        self.diffusion_model = None
        self.diffusion_seed = None  
        self.device = device
        self.seed = None # random seed for training
        self.eval_seed = None
        self.synthetic_dataset = None
        self.epochs = None
        self.params = params
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

    def get_alpha(self, x):
        return round(self.kd_scheduling(x), 2)

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

    def init_client(self):
        """
        Initialize the client model training process
        """
        # torch.manual_seed(self.seed)
        print(f"Initializing client {self.id}")

        # Log model properties
        # self.logger.add_text()

        torch.manual_seed(self.id)
        # self.model = self.model(weights=None, num_classes=self.params["num_classes"])
        self.model = self.model(self.params["num_classes"])
        if torch.cuda.device_count() > 1:
            print("Using multiple GPUs")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.optimizer = self.params["optimizer"](self.model.parameters(),
                                             lr=self.params["lr"], 
                                             momentum=self.params["momentum"],
                                             weight_decay=self.params["weight_decay"])
        self.criterion = self.params["criterion"]().to(self.device)
        self.train()

    def train(self):
        """
        Train the client model
        """
        # self.model.to(self.device) 
        torch.manual_seed(self.id)
        self.model.train()
        self.round += 1

        num_epochs = self.params["epochs"] * 2.5 if round == 0 else self.params["epochs"]

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
                    
            # Log statistics
            self.logger.add_scalar(f"Training_Loss/Client_{self.id:02}", total_loss/len(self.dataloader), self.round * self.params["epochs"] + epoch)
            self.logger.add_scalar(f"Training_Accuracy/Client_{self.id:02}", 100*total_correct/total, self.round * self.params["epochs"] + epoch)
        
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
        
        torch.manual_seed(self.id)

        # self.model.to(self.device)
        self.model.train()

        optimizer = self.params["kd_optimizer"](self.model.parameters(), lr=self.params["kd_lr"], momentum=self.params["kd_momentum"])

        kd_criterion = self.params["kd_criterion"](self.params["kd_temperature"]).to(self.device)
        criterion = self.params["criterion"]().to(self.device)

        # Set statistic variables
        kd_total_loss = 0
        total_loss = 0

        for epoch in range(self.params["kd_epochs"]):
            kd_total_loss = 0
            cls_total_loss = 0

            for batch_idx, ((data, target), logit) in enumerate(zip(synthetic_data, server_logits)):
                logit = logit[0]
                data, target, logit = data.to(self.device), target.to(self.device), logit.to(self.device)
                
                optimizer.zero_grad()

                output = self.model(data)

                kd_loss = kd_criterion(output, logit)
                cls_loss = criterion(output, target)
                loss = (1 - self.params["kd_alpha"]) * cls_loss + self.params["kd_alpha"] * kd_loss
                
                # Adaptive loss
                # max_loss = torch.max(cls_loss, kd_loss)
                # min_loss = torch.min(cls_loss, kd_loss)
                # loss_sum = cls_loss + kd_loss
                # loss = (max_loss / loss_sum) * cls_loss + (min_loss / loss_sum) * kd_loss

                kd_total_loss += kd_loss.item()
                cls_total_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Log statistics
            self.logger.add_scalar(f"KD_Loss/Client_{self.id:02}", kd_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)
            self.logger.add_scalar(f"KD_Class_Loss/Client_{self.id:02}", cls_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)

        self.logger.flush()
        del optimizer, kd_criterion, criterion


    def train_kd(self, teacher_logits, synthetic_data=None, diffusion_seed=None):
        torch.manual_seed(self.id)
        self.model.train()
        self.round += 1

        kd_criterion = self.params["kd_criterion"](self.params["kd_temperature"]).to(self.device)
        
        for epoch in range(self.params["epochs"]):
            # Set statistic variables
            kd_total_loss = 0
            cls_total_loss = 0
            total_loss = 0
            total_correct = 0
            total = 0
            for batch_idx, ((local_data, local_target), (kd_data, kd_target), logit) in enumerate(zip(self.dataloader, synthetic_data, teacher_logits)):
                # Send data and target to device
                local_data, local_target = local_data.to(self.device), local_target.to(self.device)
                logit = logit[0]
                kd_data, kd_target, logit = kd_data.to(self.device), kd_target.to(self.device), logit.to(self.device)

                # Forward pass
                # Local data training
                local_output = self.model(local_data)
                local_loss = self.criterion(local_output, local_target)

                # Synthetic data knowledge distillation
                kd_output = self.model(kd_data)
                kd_loss = kd_criterion(kd_output, logit)
                kd_cls_loss = self.criterion(kd_output, kd_target)
                
                # Adaptive knowledge distillation loss
                # max_loss = torch.max(local_loss, kd_loss)
                # min_loss = torch.min(local_loss, kd_loss)
                # loss_sum = local_loss + kd_loss
                # loss = (max_loss / loss_sum) * local_loss + (min_loss / loss_sum) * kd_loss
                
                # Knowledge distillation alpha scheduling
                if self.kd_scheduling is not None: 
                    alpha = self.get_alpha(self.round)
                    loss = (1 - alpha) * local_loss + alpha * kd_loss
                else:
                    loss = (1 - self.params["kd_alpha"]) * local_loss + self.params["kd_alpha"] * kd_loss
                    # loss = (1 - self.params["kd_alpha"]) * local_loss + self.params["kd_alpha"] * kd_loss

                # Zero out gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total += local_target.size(0)
                total_loss += local_loss.item()
                _, predicted = torch.max(local_output.data, 1)
                total_correct += (predicted == local_target).sum().item()
                kd_total_loss += kd_loss.item()
                cls_total_loss += kd_cls_loss.item()

            # Log statistics
            self.logger.add_scalar(f"Training_Loss/Client_{self.id:02}", total_loss/len(self.dataloader), self.round * self.params["epochs"] + epoch)
            self.logger.add_scalar(f"Training_Accuracy/Client_{self.id:02}", 100*total_correct/total, self.round * self.params["epochs"] + epoch)
            self.logger.add_scalar(f"KD_Loss/Client_{self.id:02}", kd_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)
            self.logger.add_scalar(f"KD_Class_Loss/Client_{self.id:02}", cls_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)

        self.logger.flush()

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

    def save_checkpoint(self):
        """
        Save the client model checkpoint
        """
        torch.save({
            'round': self.round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.checkpoint_path + f"/client_{self.id:02}.pt")
        
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