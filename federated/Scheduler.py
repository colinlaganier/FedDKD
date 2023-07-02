import torch
import Server, Client

class Scheduler:

    def __init__(self, num_clients, num_devices, server_model, client_models, dataset):
        # self.clients = clients # list of Clients
        # self.server = server
        self.num_devices = num_devices
        self.num_clients = num_clients
        
        self.server_device = None
        self.server_model = server_model

        self.client_devices = None
        self.client_models = client_models
        self.client_optimizer = client_optimizer
        self.criterion = criterion
        self.dataset = dataset

        # Training parameters
        self.round = 0
        self.num_rounds = None
        self.train_epochs = None

        # Setup
        self.assign_devices()
        self.setup_clients()
        self.server = Server(self.server_device, self.server_model, self.server_optimizer)

    def assign_devices(self, static_server=False):
        """
        Assigns GPU device id to clients
        """
        # Multi GPU 
        if (self.num_devices > 1):
            # Find optimal device assignment for balance
            num_devices_avail = self.num_devices
            if static_server: 
                num_devices_avail -= 1
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
            # Assign devices to clients
            # for i in range(len(indices)):
            #     self.clients[i].set_device(indices[i]) 
        # Single GPU
        else:
            # for i in range(self.num_clients):
            #     self.clients[i].set_device()
            self.server_device = 0
            self.client_devices = [0] * self.num_clients

    def setup_clients(self):
        clients = [None] * self.num_clients
        for client_id in range(self.num_clients):            
            clients[client_id] = Client(client_id, 
                                        self.client_devices[client_id], 
                                        self.client_models[client_id], 
                                        self.client_optimizer, 
                                        self.dataset.client_dataloaders[client_id])        

    def init_client(self):
        # train each client on local data 
        pass

    def train(self):
        self.round += 1
        # train each client in parallel
        for client in self.clients:
            client.train()

