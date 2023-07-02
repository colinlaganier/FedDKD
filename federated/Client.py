import torch
from diffusion import DiT

class Client:

    def __init__(self, client_id):
        self.id = client_id
        self.dataloader = None
        self.model = None
        # self.model = deepcopy(model).to(self.gpu_id)
        self.criterion = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.diffusion_model = None
        self.diffusion_seed = None  
        self.device = None

    def set_device(self, device):
        """
        Set the device for the client model
        
        Args:
            device (torch.device): device to set the model to
        """
        self.device = device
        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=0.001)


    def init_model(self):
        """
        Initialize the client model training process
        """
        # set learnign rate and stuff
        self.model = self.model()

    def set_dataset(self, dataset):
        """
        Set the dataset for the client model

        Args:
            dataset (torch.utils.data.DataLoader): dataset to set the model to
        """
        self.dataloader = dataset

    def init_diffusion(self):
        """
        Initialize the diffusion model training process
        """
        pass

    def get_diffusion(self):
        pass

    def generate_diffusion(self):
        pass

    def train(self, num_epoch):
        for epoch in range(num_epoch):
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

    @torch.inference_mode()
    def evaluate(self):
        pass

    def update(self):
        pass

    def knowledge_distillation(self):
        pass

    def send_logits(self):
        pass

    def get_update(self, logits, seed):
        pass