import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

class Dataset:

    def __init__(self, data_path, dataset_id, image_size, batch_size, num_clients):
        self.data_path = data_path
        self.image_size = image_size
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.equal_split = True
        self.client_dataloaders = []
        self.test_dataloader = None

        # Dataset transforms
        self.mean, self.std = self.get_stats(dataset_id)
        self.train_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_stats(self, dataset_id):
        """
        Returns mean and standard deviation of dataset
        
        Args:
            dataset_id (str): dataset identifier
        Returns:
            mean (list): mean of dataset
            std (list): standard deviation of dataset
        """
        if (dataset_id == "cifar10"):
            return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif (dataset_id == "cifar100"):
            return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        
        
    def split_data(self, training_data):
        """
        Splits training data into client datasets based on num_clients and splitting strategy
        
        Args:
            training_data (torch.utils.data.Dataset): training data
        Returns:
            client_data (list): list of client datasets
        """
        if self.equal_split:
            return random_split(training_data, [len(training_data) // self.num_clients] * self.num_clients)


    def prepare_data(self):
        """
        Loads data from data_path and splits into client and test dataloader
        """
        # Load image data and split into client datasets
        training_data = ImageFolder(self.data_path + "/train", transform=self.train_transform)
        client_data = self.split_data(training_data)
        test_data = ImageFolder(self.data_path + "/test", transform=self.test_transform)
        
        # Create client dataloaders
        for client in client_data:
            self.client_dataloaders.append(DataLoader(client, batch_size=self.batch_size, shuffle=True))
        
        # Create test dataloader
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

