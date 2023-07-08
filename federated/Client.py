import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from diffusion import DiT
from knowledge_distillation import Logits, SoftTarget

class Client:

    def __init__(self, client_id, device, model, optimizer, dataloader):
        self.id = client_id
        self.dataloader = dataloader
        self.model = model
        # self.model = deepcopy(model).to(self.gpu_id)
        self.criterion = None
        self.optimizer = None
        self.diffusion_model = None
        self.diffusion_seed = None  
        self.device = device
        self.seed = None # random seed for training

        self.epochs = None
        self.kd_epochs = None
        self.kd_temperature = 1
        self.kd_alpha = 0.5


    def set_device(self, device):
        """
        Set the device for the client model
        
        Args:
            device (torch.device): device to set the model to
        """
        self.device = device
        self.model.to(self.device)

    def init_client(self, params):
        """
        Initialize the client model training process
        """
        # torch.manual_seed(self.seed)
        torch.manual_seed(self.client_id)
        self.model = self.model(params["num_classes"])
        self.model.to(self.device)
        self.optimizer = params["optimizer"](self.model.parameters(),
                                             lr=params["lr"], 
                                             momentum=params["momentum"],
                                             weight_decay=params["weight_decay"])
        self.criterion = params["criterion"]
        self.train()

    def train(self, num_epoch):
        """
        Train the client model

        Args:
            num_epoch (int): number of epochs to train for
        """
        # self.model.to(self.device) 
        self.model.train()

        for epoch in range(self.epochs):
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

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.dataloader.dataset),
                        100. * batch_idx / len(self.dataloader), loss.item()))

    def knowledge_distillation(self, server_logits, synthetic_data=None, diffusion_seed=None):
        """
        Knowledge distillation from server to client

        Args:
            server_logits (torch.Tensor): logits from the server model
            synthetic_data (torch.utils.data.DataLoader): synthetic diffusion data - if not generated at runtime
            diffusion_seed (int): random seed for diffusion sampling - if generated at runtime
        Returns:
            torch.Tensor: logits from the client model
        """

        # Generate synthetic data if not provided
        if synthetic_data is None:
            synthetic_data = self.generate_diffusion(diffusion_seed)

        # self.model.to(self.device)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        kd_criterion = SoftTarget(self.kd_temperature).to(self.device)
        criterion = nn.CrossEntrophyLoss().to(self.device)

        for epoch in range(self.kd_epochs):
            for batch_idx, (data, target) in enumerate(synthetic_data):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                output = self.model(data)

                # loss = self.kd_alpha * kd_criterion(output, server_logits, target)
                loss = (1 - self.kd_alpha) * criterion(output, target) + \
                        self.kd_alpha * kd_criterion(output, server_logits, target)

                loss.backward()
                optimizer.step()


    def generate_logit(self, diffusion_seed=None):
        """
        Generate logits from the client model

        Args:
            diffusion_seed (int): random seed for diffusion sampling - if generated at runtime
        Returns:
            torch.Tensor: logits from the client model
        """

        if synthetic_data is None:
            synthetic_data = self.generate_diffusion(diffusion_seed)

        # self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = []

            for batch_idx, (data, target) in enumerate(synthetic_data):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                logits.append(output)

        return torch.cat(logits, dim=0) 
        # return torch.cat(logits).detach().cpu()



    @torch.inference_mode()
    def evaluate(self):
        pass

    def get_update(self, logits, seed):
        pass