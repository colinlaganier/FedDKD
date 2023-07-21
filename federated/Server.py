import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

class Server:
    
    def __init__(self, device, model, params, checkpoint_path, logger):
        # Server properties
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.seed = 42
        self.round = 0

        # Server model properties
        self.model = model
        self.params = params
        self.optimizer = None
        self.synthetic_dataset = None
        self.client_logits = None

    def init_server(self, synthetic_dataset = None, pre_train=False):
        """
        Initialize the server model training process

        Args:
            pre_train (bool): whether to pre-train the server model
        """
        print("Initializing server model")
        torch.manual_seed(self.seed)

        # self.model = self.model(weights=None, num_classes=self.params["num_classes"])
        self.model = self.model(self.params["num_classes"])
        self.model.to(self.device)
        self.optimizer = self.params["optimizer"](self.model.parameters(),
                                                lr=self.params["lr"], 
                                                momentum=self.params["momentum"],
                                                weight_decay=self.params["weight_decay"])
        self.criterion = self.params["criterion"]().to(self.device)

        if pre_train:
            self.synthetic_train(synthetic_dataset)

    def aggregate_logits(self, client_logits):
        """
        Aggregate client logits into a single tensor of average logits

        Args:
            client_logits (Queue): queue of client logits
        """

        num_clients = client_logits.qsize()
        logit_sum = torch.zeros_like(client_logits.queue[0])
        
        # Aggregate client logits
        while not client_logits.empty():
            logit_sum += client_logits.get()

        # Put average logit value  back into queue
        client_logits.put(logit_sum / num_clients)
        self.client_logits = client_logits

    def knowledge_distillation(self, synthetic_data=None, diffusion_seed=None):
        """
        Knowledge distillation from client logits to server model
        
        Args:
            synthetic_data (TensorDataset): synthetic diffusion data
            diffusion_seed (int): random seed for diffusion
        """
        if synthetic_data is None:
            synthetic_data = self.synthetic_dataset

        torch.manual_seed(self.seed)
        self.model.train()
        self.round += 1

        while not self.client_logits.empty():
            # client_logit = logit_queue.get()
            client_logit = DataLoader(TensorDataset(self.client_logits.get()), batch_size=self.params["kd_batch_size"], num_workers=4)
            optimizer = self.params["kd_optimizer"](self.model.parameters(), lr=self.params["kd_lr"], momentum=self.params["kd_momentum"])

            kd_criterion = self.params["kd_criterion"](self.params["kd_temperature"]).to(self.device)
            criterion = self.params["criterion"]().to(self.device)

            for epoch in range(self.params["kd_epochs"]):
                kd_total_loss = 0
                cls_total_loss = 0

                for batch_idx, ((data, target), logit) in enumerate(zip(synthetic_data, client_logit)):
                    logit = logit[0]
                    data, target, logit = data.to(self.device), target.to(self.device), logit.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    output = self.model(data)

                    # Compute loss
                    kd_loss = kd_criterion(output, logit)
                    cls_loss = criterion(output, target)
                    loss = (1 - self.params["kd_alpha"]) * cls_loss + self.params["kd_alpha"] * kd_loss
                    
                    kd_total_loss += kd_loss.item()
                    cls_total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                
                # Log statistics
                self.logger.add_scalar("KD_Loss/Server", kd_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)
                self.logger.add_scalar("KD_Class_Loss/Server", cls_total_loss/len(synthetic_data), self.round * self.params["kd_epochs"] + epoch)

            self.logger.flush()
            del optimizer, kd_criterion, criterion

    def generate_logits(self, synthetic_data=None, diffusion_seed=None):
        """
        Generate logits from the server model

        Args:
            synthetic_data (TensorDataset): synthetic diffusion data
            diffusion_seed (int): random seed for diffusion sampling - if generated at runtime
        Returns:
            torch.Tensor: logits from the client model
        """

        torch.manual_seed(self.seed)

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

        return torch.cat(logits).detach().cpu()


    @torch.inference_mode()
    def evaluate(self, dataloader):
        """
        Evaluate the client model on the test dataset

        Args:
            dataloader (DataLoader): test dataset
        """
        torch.manual_seed(self.params["eval_seed"])
        self.model.eval()
        
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total = 0 
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total += target.size(0)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()

         # Log statistics
            self.logger.add_scalar("Validation_Loss/Server", total_loss/len(dataloader), self.round)
            self.logger.add_scalar("Validation_Accuracy/Server",  100*total_correct/total, self.round)
            self.logger.flush()

    def synthetic_train(self, synthetic_dataset):
        """
        Train the server model on synthetic data for initialisation
        """

        torch.manual_seed(self.seed)

        self.model.train()
        for epoch in range(25):
            total_loss = 0
            total_correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(synthetic_dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total += target.size(0)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()

            # Log statistics
            self.logger.add_scalar("Training_Loss/Server", total_loss/len(synthetic_dataset), self.round * self.params["epochs"] + epoch)
            self.logger.add_scalar(f"Training_Accuracy/Server", 100*total_correct/total, self.round * self.params["epochs"] + epoch)
            self.logger.flush() 

    def save_checkpoint(self):
        """
        Save the server model checkpoint
        """
        torch.save({
            'round': self.round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.checkpoint_path + f"/server.pt")
        
    def load_checkpoint(self, checkpoint):
        """
        Load the client model checkpoint

        Args:
            checkpoint (str): path to checkpoint
        """
        checkpoint = torch.load(checkpoint)
        self.round = checkpoint['round']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])