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
    def resnet(num_clients):
        return [Models.ResNet18] * (num_clients // 2) + [Models.ResNet34] * (num_clients - (num_clients // 2))
    
    @staticmethod
    def strategy_2(num_clients):
        return [Models.MobileNetV3] * (num_clients // 2) + [Models.ShuffleNetV2] * (num_clients - (num_clients // 2))
    
    @staticmethod
    def cnn_1(num_clients):
        return [Models.CNN_small]  + [Models.CNN_medium] * 2 + [Models.CNN_large] * (num_clients - 3)
    
    available = {"heterogeneous_random" : heterogeneous_random,
                 "homogeneous_random" : heterogeneous_random,
                 "homogenous" : homogenous,
                 "resnet" : resnet,
                 "strategy_2" : strategy_2,
                 "cnn_1": cnn_1}