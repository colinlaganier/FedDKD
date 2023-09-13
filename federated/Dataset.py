import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import EMNIST

class Dataset:

    def __init__(self, data_path, dataset_id, batch_size, kd_batch_size, num_clients, synthetic_path=None):
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.synthetic_path = synthetic_path
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.kd_batch_size = kd_batch_size
        self.equal_split = True
        self.client_dataloaders = []
        self.client_pretrain_dataloader = []
        self.test_dataloader = None
        self.num_classes = 10
        self.synthetic_dataset = []
        self.server_synthetic_dataset = None

        # Dataset transforms
        self.mean, self.std = self.get_stats(dataset_id)
        self.image_size = self.get_image_size(dataset_id)
        if (dataset_id == "cinic10"):
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        elif (dataset_id == "emnist"):
            self.train_transform = transforms.Compose([
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])]
                )
            self.synthetic_transform = transforms.Compose([
                transforms.ToTensor(),
                lambda img: img[0,:,:].unsqueeze(0),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.diffusion_transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
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
        elif (dataset_id == "emnist"):
            return [0.5], [0.5]
        
    def get_image_size(self, dataset_id):
        if (dataset_id in ["cifar10", "cifar100", "cinic10"]):
            return 32   
        elif dataset_id == "emnist":
            return 32
        
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
        np.random.seed(42)
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


    def prepare_data(self, partition, load_diffusion):
        """
        Loads data from data_path and splits into client and test dataloader

        Args:
            partition (str): partition strategy for splitting training data
            load_diffusion (bool): whether to load pre-generated synthetic data
        """
        
        if self.dataset_id == "emnist":
            # Load EMNIST dataset from torchvision
            training_data = EMNIST(root='./dataset', train=True, download=True, transform=self.train_transform, split='digits')
            test_data = EMNIST(root='./dataset', train=False, download=True, transform=self.train_transform, split='digits')

            # Reduce test set to 10,000 images for consistency
            test_split = self.balanced_split(test_data, 4)
            test_data = test_split[0]

            if load_diffusion:
                assert self.synthetic_path is not None, "Synthetic path must be specified for loading synthetic emnist"
                print("Loading synthetic data from {}".format(self.synthetic_path))
                # Load pre-generated synthetic dataset
                synthetic_data = ImageFolder(self.synthetic_path, transform=self.synthetic_transform)
                print("Synthetic data size: {}".format(len(synthetic_data)))
                self.synthetic_dataset = self.balanced_split(synthetic_data, 10)
                print("Synthetic dataset size per round: {}".format(len(self.synthetic_dataset[0])))
            else:
                # Use EMNIST test set as synthetic dataset
                self.synthetic_dataset = test_split[1:]

        elif self.dataset_id == "cinic10":
            training_data = ImageFolder(self.data_path + "/train", transform=self.train_transform)
            test_data = ImageFolder(self.data_path + "/test", transform=self.test_transform)

            # Reduce test set to 10,000 images for consistency
            test_data = self.balanced_split(test_data, 9)[0]

            if load_diffusion:
                if self.synthetic_path is not None:
                    print("Loading synthetic data from {}".format(self.synthetic_path))
                    # Load pre-generated synthetic dataset
                    synthetic_data = ImageFolder(self.synthetic_path, transform=self.test_transform)
                    self.synthetic_dataset = self.balanced_split(synthetic_data, 10)
                    print("Synthetic dataset size per round: {}".format(len(self.synthetic_dataset[0])))
                else:
                    synthetic_data = ImageFolder(self.data_path + "/valid", transform=self.test_transform)
                    self.synthetic_dataset = self.balanced_split(synthetic_data, 9)
        
        # Split training data into client datasets based on partition strategy
        if (partition == "iid"):
            client_data = self.balanced_split(training_data, self.num_clients)
        elif (partition == "random"):
            client_data = self.random_split(training_data, self.num_clients)
        elif (partition == "dirichlet"):
            client_data = self.dirichlet_split(training_data, self.num_clients)

        # Create client dataloaders
        for client in client_data:
            # self.client_pretrain_dataloader.append(DataLoader(client, batch_size=self.pretrain_batch_size, shuffle=True))
            self.client_dataloaders.append(DataLoader(client, batch_size=self.batch_size, shuffle=True, drop_last=True))
        
        for i in range(len(self.client_dataloaders)):
            print("Client {} size: {}".format(i, len(self.client_dataloaders[i].dataset)))
        
        # Create test dataloader
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    def get_synthetic_data(self, round=None):
        """
        Loads synthetic data based on round 

        Args:
            round (int): round number
        Returns:
            synthetic_dataloader (torch.utils.data.DataLoader): synthetic data loader
        """
        if self.dataset_id == "emnist":
            if round is None:
                synthetic_data = self.synthetic_dataset[0]
            else:
                if self.synthetic_path:
                    round = round % 20
                else: 
                    round = round % 3
                synthetic_data = self.synthetic_dataset[round]
        else: 
            if round is None:
                synthetic_data = self.synthetic_dataset[0]
            else:
                if self.synthetic_path:
                    round = round % 20
                else:
                    round = round % 9
                synthetic_data = self.synthetic_dataset[round]

        # Create synthetic dataloader
        synthetic_dataloader = DataLoader(synthetic_data, batch_size=self.kd_batch_size, shuffle=False)
        return synthetic_dataloader