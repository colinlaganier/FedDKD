import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from diffusion import DiT
from knowledge_distillation import Logits, SoftTarget

class Client:

    def __init__(self, client_id, device, model, dataloader, params, logger):
        # Client properties
        self.id = client_id
        self.logger = logger
        self.checkpoint_path = "checkpoints/"
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
        # self.kd_epochs = None
        # self.kd_temperature = 1
        # self.kd_alpha = 0.5
        self.params = params

    def set_device(self, device):
        """
        Set the device for the client model
        
        Args:
            device (torch.device): device to set the model to
        """
        self.device = device
        self.model.to(self.device)

    def init_client(self):
        """
        Initialize the client model training process
        """
        # torch.manual_seed(self.seed)
        print(f"Initializing client {self.id}")

        # Log model properties
        # self.logger.add_text()

        torch.manual_seed(self.id)
        self.model = self.model(self.params["num_classes"])
        self.model.to(self.device)
        self.optimizer = self.params["optimizer"](self.model.parameters(),
                                             lr=self.params["lr"], 
                                             momentum=self.params["momentum"],
                                             weight_decay=self.params["weight_decay"])
        self.criterion = self.params["criterion"]().to(self.device)
        # self.train()

    def train(self):
        """
        Train the client model
        """
        # self.model.to(self.device) 
        torch.manual_seed(self.id)
        self.model.train()
        self.round += 1

        for epoch in range(self.params["epochs"]):
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
                
                kd_total_loss += kd_loss.item()
                cls_total_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Log statistics
            self.logger.add_scalar(f"KD_Loss/Client_{self.id:02}", kd_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)
            self.logger.add_scalar(f"KD_Class_Loss/Client_{self.id:02}", cls_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)

        self.logger.flush()
        del optimizer, kd_criterion, criterion


    def generate_logits(self, synthetic_data=None, diffusion_seed=None):
        """
        Generate logits from the client model

        Args:
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
            round = self.round + 0.5 if post_kd else self.round
            self.logger.add_scalar(f"Validation_Loss/Client_{self.id:02}", total_loss/len(test_dataloader), round)
            self.logger.add_scalar(f"Validation_Accuracy/Client_{self.id:02}", 100*total_correct/total, round)
            self.logger.flush()

    def save_checkpoint(self):
        """
        Save the client model checkpoint
        """
        torch.save({
            'round': self.round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.checkpoint_path)
        
    def load_checkpoint(self, checkpoint):
        """
        Load the client model checkpoint
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