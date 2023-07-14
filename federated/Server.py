import torch
import torch.nn as nn
import random

class Server:
    
    def __init__(self, device, model, params, logger):
        # Server properties
        self.device = device
        self.logger = logger
        self.seed = 42
        self.round = 0

        # Server model properties
        self.model = model
        self.params = params
        self.optimizer = None
        self.synthetic_dataset = None

    def init_server(self):
        self.model = self.model(self.params["num_classes"])
        self.model.to(self.device)
        self.optimizer = self.params["optimizer"](self.model.parameters(),
                                                lr=self.params["lr"], 
                                                momentum=self.params["momentum"],
                                                weight_decay=self.params["weight_decay"])
        self.criterion = self.params["criterion"]().to(self.device)

        # if pre_train:
        #     self.synthetic_train()

    def knowledge_distillation(self, logit_queue, synthetic_data=None, diffusion_seed=None):
        """
        Knowledge distillation from client logits to server model
        
        Args:
            logit_queue (Queue): queue of client logits
            synthetic_data (TensorDataset): synthetic diffusion data
            diffusion_seed (int): random seed for diffusion
        """
        if synthetic_data is None:
            synthetic_data = self.synthetic_dataset

        torch.manual_seed(self.seed)
        self.model.train()
        self.round += 1

        while not logit_queue.empty():
            client_logit = logit_queue.get()

            optimizer = self.params["kd_optimizer"](self.model.parameters(), lr=self.params["kd_lr"], momentum=self.params["kd_momentum"])

            kd_criterion = self.params["kd_criterion"](self.params["kd_temperature"]).to(self.device)
            criterion = self.params["criterion"]().to(self.device)

            for epoch in range(self.params["kd_epochs"]):
                for batch_idx, ((data, target), logit) in enumerate(zip(synthetic_data, client_logit)):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)

                    # Compute loss
                    kd_loss = self.params["kd_alpha"] * kd_criterion(output, logit)
                    loss = (1 - self.params["kd_alpha"]) * criterion(output, target) + self.params["kd_alpha"] * kd_loss
                    
                    kd_total_loss += kd_loss.item()
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                
                # Log statistics
                self.logger.add_scalar("KD_Loss/Server", kd_total_loss/len(synthetic_data), epoch)
                self.logger.add_scalar("KD_Total_Loss/Server", total_loss/len(synthetic_data), epoch)

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
        torch.manual_seed(self.eval_seed)
        self.model.eval()
        
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * len(data)
                total_correct += output.argmax(dim=1).eq(target).sum().item() * len(data)

         # Log statistics
            self.logger.add_scalar("Validation_Loss/Server", total_loss/len(dataloader), self.round)
            self.logger.add_scalar("Validation_Accuracy/Server", total_correct/len(dataloader), self.round)
            self.logger.flush()

    def synthetic_train(self):
        """
        Train the server model on synthetic data for initialisation
        """

        torch.manual_seed(self.seed)

        self.model.train()
        print("Training on synthetic data...")
        for epoch in range(self.params["synthetic_epochs"]):
            print(f"Epoch {epoch+1}/{self.params['synthetic_epochs']}")
            for batch_idx, (data, target) in enumerate(self.synthetic_dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                print(f"Loss: {loss.item()}")

            # Log statistics
            self.logger.add_scalar("Loss/Server", loss.item(), epoch)
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