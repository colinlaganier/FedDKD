import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split

class Dataset:

    def __init__(self, data_path, dataset_id, batch_size, kd_batch_size, num_clients, synthetic_path=None):
        self.data_path = data_path
        self.synthetic_path = synthetic_path
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.kd_batch_size = kd_batch_size
        self.equal_split = True
        self.client_dataloaders = []
        self.test_dataloader = None
        self.num_classes = 10 if (dataset_id == "cifar10") else 100
        # self.diffusion

        # Dataset transforms
        self.mean, self.std = self.get_stats(dataset_id)
        self.train_transform = transforms.Compose([
            # transforms.Resize((32,32)),
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
        self.diffusion_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
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
        elif (dataset_id == "cinic10"):
            return [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
        
        
    def random_split(self, dataset, num_splits):
        """
        Splits training data into client datasets based on num_clients and splitting strategy
        
        Args:
            dataset (torch.utils.data.Dataset): training data
            num_splits (int): number of client datasets to split into
        Returns:s
            client_data (list): list of client datasets
        """
        if self.equal_split:
            return random_split(dataset, [len(dataset) // num_splits] * num_splits)
        
    def balanced_split(self, dataset, num_splits):
        """
        Splits training data into client datasets with balanced classes
        
        Args:
            dataset (torch.utils.data.Dataset): training data
            num_splits (int): number of client datasets to split into
        Returns:
            client_data (list): list of client datasets
        """
        samples_per_class = len(dataset) // num_splits
        remainder = len(dataset) % num_splits

        class_counts = [0] * self.num_classes # number of samples per class
        subset_indices = [[] for _ in range(num_splits)] # indices of samples per subset
        for i, (data, target) in enumerate(dataset):
            # Add sample to subset if number of samples per class is less than samples_per_class
            if class_counts[target] < samples_per_class:
                subset_indices[i % num_splits].append(i)
                class_counts[target] += 1
            elif remainder > 0:
                subset_indices[i % num_splits].append(i)
                class_counts[target] += 1
                remainder -= 1

        # Create Subset objects for each subset
        subsets = [Subset(dataset, indices) for indices in subset_indices]

        return subsets
        

    def dirichlet_split(self, dataset, num_splits, beta=0.1):
        """
        Splits training data into client datasets based Dirichlet distribution

        Args:
            dataset (torch.utils.data.Dataset): training data
            num_splits (int): number of client datasets to split into
            beta (float): concentration parameter of Dirichlet distribution
        Returns:
            client_data (list): list of client datasets       
        """

        label_distributions = []
        # Generate label distributions for each class using Dirichlet distribution
        for y in range(len(dataset.classes)):
            label_distributions.append(np.random.dirichlet(np.repeat(beta, num_splits)))

        labels = np.array(dataset.targets).astype(int)
        client_idx_map = {i: {} for i in range(num_splits)}
        client_size_map = {i: {} for i in range(num_splits)}

        for y in range(len(dataset.classes)):
            label_y_idx = np.where(labels == y)[0]
            label_y_size = len(label_y_idx)

            # Sample number of samples for each client from label distribution
            sample_size = (label_distributions[y] * label_y_size).astype(int)
            sample_size[num_splits - 1] += len(label_y_idx) - np.sum(sample_size)
            for i in range(num_splits):
                client_size_map[i][y] = sample_size[i]

            np.random.shuffle(label_y_idx)
            sample_interval = np.cumsum(sample_size)
            for i in range(num_splits):
                client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]

        subsets = []
        for i in range(num_splits):
            client_i_idx = np.concatenate(list(client_idx_map[i].values()))
            np.random.shuffle(client_i_idx)
            subsets.append(Subset(dataset, client_i_idx))

        return subsets


    def prepare_data(self, partition):
        """
        Loads data from data_path and splits into client and test dataloader
        """
        # Load image data and split into client datasets
        training_data = ImageFolder(self.data_path + "/train", transform=self.train_transform)
        # Split training data into client datasets based on partition strategy
        if (partition == "iid"):
            client_data = self.balanced_split(training_data, self.num_clients)
        elif (partition == "random"):
            client_data = self.random_split(training_data, self.num_clients)
        elif (partition == "dirichlet"):
            client_data = self.dirichlet_split(training_data, self.num_clients)

        test_data = ImageFolder(self.data_path + "/test", transform=self.test_transform)
        
        # Create client dataloaders
        for client in client_data:
            self.client_dataloaders.append(DataLoader(client, batch_size=self.batch_size, shuffle=True))
        
        # Create test dataloader
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    # def prepare_diffusion_data(self):
    #     """
    #     Loads data from data_path and splits into client dataloader
    #     """
    #     diffusion_data = ImageFolder(self.data_path + "/diffusion", transform=self.diffusion_transform)
    #     self.diffusion_dataloader = DataLoader(diffusion_data, batch_size=self.batch_size, shuffle=True)

    def set_synthetic_data(self):
        self.synthetic_folder = self.data_path + "/synthetic"

    def get_synthetic_data(self, round=None):
        """
        Loads synthetic data from synthetic_folder
        """
        if round is None:
            synthetic_data = ImageFolder(self.synthetic_path, transform=self.train_transform)
        else:
            synthetic_data = ImageFolder(self.synthetic_path + "/round_" + str(round), transform=self.train_transform)
        synthetic_dataloader = DataLoader(synthetic_data, batch_size=self.kd_batch_size, shuffle=True)
        return synthetic_dataloader