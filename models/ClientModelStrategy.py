import random
from models.Models import Models

class ClientModelStrategy:

    @staticmethod
    def heterogeneous_random(num_clients):
        return random.choices(list(Models.available.values()), k=num_clients)
    
    @staticmethod
    def homogeneous_random(num_clients):
        return [random.choice(list(Models.available.values()))] * num_clients
    
    @staticmethod
    def homogenous(num_clients, model):
        return [model] * num_clients
    
    @staticmethod
    def strategy_1(num_clients):
        return [Models.ResNet18] * (num_clients // 2) + [Models.ResNet32] * (num_clients - (num_clients // 2))
    
    @staticmethod
    def strategy_2(num_clients):
        return [Models.MobileNetV3] * (num_clients // 2) + [Models.ResNet18] * (num_clients - (num_clients // 2))
    
    available = {"heterogeneous_random" : heterogeneous_random,
                 "homogeneous_random" : heterogeneous_random,
                 "homogenous" : homogenous,
                 "strategy_1" : heterogeneous_random,
                 "strategy_2" : heterogeneous_random}