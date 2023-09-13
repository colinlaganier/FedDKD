import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from queue import Queue
from multiprocessing import Queue as mpQueue
from federated.Client import Client
from federated.Dataset import Dataset
from federated.Server import Server
from models.Models import Models
from knowledge_distillation import Logits, SoftTarget

from federated.Diffusion import load_model
from federated.diffusion_utils import sample

class Scheduler:

    def __init__(self, 
                 num_devices, 
                 num_clients,
                 server_model, 
                 client_models,
                 epochs,
                 kd_epochs, 
                 batch_size, 
                 kd_batch_size,
                 data_path, 
                 dataset_id, 
                 data_partition, 
                 synthetic_path,
                 load_diffusion,
                 save_checkpoint,
                 checkpoint_path,
                 logger):

        self.logger = logger
        self.save_checkpoint = save_checkpoint
        self.checkpoint_path = checkpoint_path
        self.load_diffusion = load_diffusion

        self.clients = [None] * num_clients
        self.device_dict = dict(zip(range(num_devices), [[] for _ in range(num_devices)]))
        self.num_devices = num_devices
        self.num_clients = num_clients
        self.server_device = None
        self.server_model = Models.available[server_model]
        self.client_devices = None
        self.client_models = client_models

        # Training parameters
        self.round = 0
        self.train_epochs = epochs
        self.kd_epochs = kd_epochs
        self.batch_size = batch_size
        self.kd_batch_size = kd_batch_size
        self.eval_seed = random.randint(0, 100000)

        # model and dataset dependant
        self.training_params = {"num_classes": 10,
                                "optimizer": optim.SGD, 
                                "criterion": nn.CrossEntropyLoss, 
                                "lr": 0.01 , 
                                "momentum": 0.9, 
                                "epochs": self.train_epochs,
                                "weight_decay": 1e-4, 
                                "batch_size": self.batch_size,
                                "kd_optimizer": optim.SGD,
                                "kd_criterion": SoftTarget,
                                "kd_lr": 0.01,
                                "kd_momentum": 0.9,
                                "kd_alpha": 0.8,
                                "kd_alpha_server": 0.8,
                                "kd_temperature": 4,
                                "kd_epochs": self.kd_epochs,
                                "kd_epochs_server": 5,
                                "pretrain_epochs": 20,
                                "kd_batch_size": self.kd_batch_size,
                                "eval_seed": self.eval_seed,
                                "kd_scheduling": None}

        # Setup datasets
        self.dataset_id = dataset_id
        self.dataset = Dataset(data_path, dataset_id, batch_size, kd_batch_size, num_clients, synthetic_path)
        self.dataset.prepare_data(data_partition, load_diffusion)
        
        # Assign devices to server and clients based on number of devices
        self.assign_devices()

        # Loading Diffusion Model
        self.diffusion_model = None
        if not load_diffusion:
            self.init_DM(synthetic_path)

        # Setup server and initialize
        self.server = Server(self.server_device, self.server_model(), self.dataset_id, self.training_params, self.checkpoint_path, self.logger)
        self.server.init_server()

        # Setup clients and initialize
        self.setup_clients()
        self.init_clients()

    def init_DM(self, synthetic_path):
        # Load Diffusion Model weights
        checkpoint = torch.load(synthetic_path)

        # Initialize Diffusion Model
        self.diffusion_model = load_model(1)
        self.diffusion_model.load_state_dict(checkpoint)
        self.diffusion_model.to("cuda:0")

    def assign_devices(self):
        """
        Assigns GPU device id to clients
        """
        self.server_device = 0
        self.client_devices = [0] * self.num_clients

    def setup_clients(self):
        """
        Initialize client objects
        """
        for client_id in range(self.num_clients):            
            self.clients[client_id] = Client(client_id, 
                                        self.client_devices[client_id], 
                                        self.client_models[client_id](), 
                                        self.dataset.client_dataloaders[client_id],
                                        self.dataset_id,
                                        self.training_params,
                                        self.checkpoint_path,
                                        self.logger)        
            
            # Assign client to device in device_dict
            self.device_dict[self.client_devices[client_id]].append(client_id)

    def init_clients(self):
        # train each client on local data 
        for client in self.clients:
            client.init_client()
            client.evaluate(self.dataset.test_dataloader)
        if self.save_checkpoint:
            self.save_checkpoints()

    def train_baseline(self, num_epochs=20):
        for client in self.clients:
            print("Client {}".format(client.id))
            for _ in range(num_epochs):
                client.train(epochs=1)
                client.evaluate(self.dataset.test_dataloader)

    def train(self, num_rounds, logit_ensemble=True):
        """
        Train client models on local data and perform co-distillation
        
        Args:
            num_rounds (int): number of rounds to train
            logit_ensemble (bool): whether to perform logit ensemble
        """
        # logit_queue = Queue()
        logit_arr = []
        # synthetic_dataset = self.synthetic_dataset
        diffusion_seed = None

        print("Training network for {} communication rounds".format(num_rounds))

        for round in range(num_rounds):
            print("Round {}".format(round))
            self.round += 1
            logit_arr.clear()

            if self.load_diffusion:
                synthetic_dataset = self.dataset.get_synthetic_data(round)
            else:
                seed = random.randint(0, 100000)
                synthetic_dataset = self.sample_DM(seed)

            print("Local training")
            for client in self.clients:
                # Train client on local data
                print("Client {}".format(client.id))
                client.train()
                client.evaluate(self.dataset.test_dataloader)

                # Generate logit for server update
                logit_arr.append(client.generate_logits(synthetic_dataset, diffusion_seed))

            print("Knowledge distillation")
            for idx, client in enumerate(self.clients):
                # Calculate teacher logits for client
                teacher_logits = torch.mean(torch.stack(logit_arr[:idx] + logit_arr[idx + 1:]), dim=0)
                teacher_logits = DataLoader(TensorDataset(teacher_logits), batch_size=self.kd_batch_size)

                # Update client model with teacher logits
                client.knowledge_distillation(teacher_logits, synthetic_dataset, diffusion_seed)

            client_logits = torch.mean(torch.stack(logit_arr), dim=0)
            # Update server model with client logits
            self.server.knowledge_distillation(client_logits, synthetic_dataset, diffusion_seed)
            self.server.evaluate(self.dataset.test_dataloader)

            # Save checkpoint
            if self.save_checkpoint and self.round % 5 == 0:
                self.save_checkpoints()

    def save_checkpoints(self):
        """
        Save checkpoints for server and clients
        """
        for client in self.clients:
            client.save_checkpoint()
        self.server.save_checkpoint()

    def sample_DM(self, seed): 
        """
        Sample from Diffusion Model to generate synthetic dataset

        Args:
            seed (int): seed for sampling from diffusion model

        Returns:
            synthetic_loader (DataLoader): dataloader for synthetic dataset
        """
        # Store current seed and set new seed
        current_seed = torch.seed()
        torch.manual_seed(seed)

        num_samples = 10000
        num_channels = 1
        steps = 500
        eta = 1.
        device = "cuda:0"

        # Generate random noise and class labels
        noise = torch.randn(num_samples, num_channels, 32, 32).to(device)
        fakes_classes = torch.arange(10, device=device).repeat_interleave(num_samples // 10, 0)
        
        # Sample from diffusion model
        fakes = sample(self.diffusion_model, noise, steps, eta, fakes_classes)

        synthetic_loader = DataLoader(TensorDataset(fakes, fakes_classes), batch_size=self.kd_batch_size, shuffle=False)

        # Reset seed
        torch.manual_seed(current_seed)

        return synthetic_loader