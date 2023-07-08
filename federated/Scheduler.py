import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from queue import Queue
import Client, Dataset, Server

class Scheduler:

    def __init__(self, num_clients, 
                 num_devices, 
                 server_model, 
                 client_models, 
                 epochs, 
                 batch_size, 
                 num_rounds, 
                 data_path, 
                 dataset_id, 
                 data_partition, 
                 load_diffusion):

        self.clients = [None] * num_clients
        self.device_dict = dict(zip(range(num_devices), [[] for _ in range(num_devices)]))

        self.num_devices = num_devices
        self.num_clients = num_clients
        
        self.server_device = None
        self.server_model = server_model

        self.client_devices = None
        self.client_models = client_models
        # self.client_optimizer = client_optimizer
        # self.criterion = criterion
        # model and dataset dependant
        self.training_params = {"optimizer": optim.SGD, 
                                "criterion": nn.CrossEntropyLoss, 
                                "lr": 0.1 , 
                                "momentum": 0.9, 
                                "weight_decay": 1e-4, 
                                "num_classes": 10}

        # Training parameters
        self.round = 0
        self.num_rounds = num_rounds
        self.train_epochs = epochs
        self.batch_size = batch_size

        # Setup
        self.dataset = Dataset(data_path, dataset_id, batch_size, num_clients)
        self.dataset.prepare_data(data_partition)
        if load_diffusion:
            self.dataset.set_synthetic_data()
        self.assign_devices()
        self.setup_clients()
        self.server = Server(self.server_device, self.server_model, self.server_optimizer)
        self.init_clients()

    def assign_devices(self):
        """
        Assigns GPU device id to clients
        """
        # Multi GPU 
        if (self.num_devices > 1):
            # Find optimal device assignment for balance
            num_devices_avail = self.num_devices
            avg_num_clients = (self.num_clients + 1) // num_devices_avail
            remainder = (self.num_clients + 1) % num_devices_avail

            client_devices = []
            start = 0

            for i in range(num_devices_avail):
                if remainder > 0:
                    end = start + avg_num_clients + 1
                    remainder -= 1
                else:
                    end = start + avg_num_clients

                client_devices.extend([i] * (end - start))
                start = end

            self.server_device = client_devices[0]
            self.client_devices = client_devices[1:]            
        # Single GPU
        else:
            # Assign all clients to the same device
            self.server_device = 0
            self.client_devices = [0] * self.num_clients

    def setup_clients(self):
        """
        Initialize client objects
        """
        for client_id in range(self.num_clients):            
            self.clients[client_id] = Client(client_id, 
                                        self.client_devices[client_id], 
                                        self.client_models[client_id], 
                                        self.dataset.client_dataloaders[client_id])        
            
            # Assign client to device in device_dict
            self.device_dict[self.client_devices[client_id]].append(client_id)

    def init_clients(self):
        # train each client on local data 
        for client in self.clients:
            client.init_model(self.training_params)
        pass

    def train(self, num_rounds):
        """
        Train client models on local data and perform co-distillation
        
        Args:
            num_rounds (int): number of rounds to train
            train_epochs (int): number of epochs to train each client
        """
        logit_queue = Queue()

        for round in num_rounds:
            self.round += 1

            if self.load_diffusion:
                synthetic_dataset = self.dataset.get_synthetic_data(round)
                diffusion_seed = None
                server_logit = self.server.generate_logit(diffusion_seed, synthetic_dataset)
            else:
                diffusion_seed = self.server.generate_seed()
                server_logit = self.server.generate_logit(diffusion_seed)

            

            # train each client in parallel
            for client in self.clients:
                client.train(self.train_epochs)
                if self.load_diffusion:
                    client.set_synthetic_dataset(synthetic_dataset)
                client.knowledge_distillation(server_logit, diffusion_seed)
                client_logit = client.generate_logit(diffusion_seed)
                logit_queue.put(client_logit)

            self.server.knowledge_distillation(logit_queue)


    def client_update(self, client, server_logit, diffusion_seed, logit_queue):
        # Train client on local data
        client.train(self.train_epochs)
        # Knowledge distillation with server logit and synthetic diffusion data
        client.knowledge_distillation(server_logit, diffusion_seed)
        # Generate logit for server update
        client_logit = client.generate_logit(diffusion_seed)
        logit_queue.put(client_logit)

    def train_mp(self, num_rounds, train_epochs):

        # Initialize logit queue for server update after each round
        logit_queue = Queue()

        for _ in num_rounds:
            self.round += 1

            diffusion_seed = self.server.generate_seed()
            server_logit = self.server.get_logit()
            processes = [] 

            # Start processes for each client on each device
            for i in range(math.ceil(self.num_clients / self.num_devices)):
                for device, client_ids in self.device_dict.items():
                    if i < len(client_ids):
                        process = mp.Process(target=self.client_update, args=(self.clients[client_ids[i]], server_logit, diffusion_seed, logit_queue))
                        process.start()
                        processes.append(process)

            # Wait for all processes to finish
            for process in processes:
                process.join()

            # Update server model with client logit queue
            self.server.knowledge_distillation(logit_queue)
