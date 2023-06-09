import torch

class Scheduler:

    def __init__(self, clients, server, num_devices):
        self.clients = clients # list of Clients
        self.server = server
        self.num_devices = num_devices
        self.iteration = 0

    def assign_devices(self):
        """
        Assigns GPU device id to clients
        """
        # Multi GPU 
        if (self.num_devices > 1):
            # Find optimal device assignment for balance
            num_devices_avail = self.num_devices - 1
            avg_num_clients = len(self.clients) // num_devices_avail
            remainder = len(self.clients) % num_devices_avail

            indices = []
            start = 0

            for i in range(num_devices_avail):
                if remainder > 0:
                    end = start + avg_num_clients + 1
                    remainder -= 1
                else:
                    end = start + avg_num_clients

                indices.extend([i] * (end - start))
                start = end
            
            # Assign devices to clients
            for i in range(len(indices)):
                self.clients[i].set_device(indices[i]) 
        # Single GPU
        else:
            for i in range(len(self.clients)):
                self.clients[i].set_device()

    # def add_client(self):
    #     pass

    def init_client(self):
        # train each client on local data 
        pass

    def train(self):
        self.iteration += 1

