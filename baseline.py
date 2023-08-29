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
from models.ClientModelStrategy import ClientModelStrategy
from torch.utils.tensorboard import SummaryWriter

# data_path = "dataset\\cinic-10"
# dataset_id = "cinic10"
# batch_size = 128
# kd_batch_size = 128
# num_clients = 5
# synthetic_path = "dataset\\cinic-10\\10K"
# data_partition = "iid"
# client_model = "strategy_1"



data_path = None
# dataset_id = "emnist"
batch_size = 128
kd_batch_size = 128
num_clients = 5
synthetic_path = "None"
# data_partition = "dirichlet"
# client_model = "strategy_1"

datasets = ["emnist", "cinic10"]
settings = ["iid", "dirichlet"]
models_strategy = ["cnn_1", "strategy_1"]

log_names = {"cnn_1" : "CNN", "strategy_1" : "ResNet", "emnist" : "EMNIST", "cinic10" : "CINIC-10", "iid" : "IID", "dirichlet" : "Dirichlet"}

for dataset in datasets:
    dataset_id = dataset 

    for setting in settings:
        
        data_partition = setting

        for models in models_strategy:
            
            logger = SummaryWriter(log_dir=f"runs/Baseline_{log_names[dataset_id]}_{log_names[data_partition]}_{log_names[models]}")

            dataset = Dataset(data_path, dataset_id, batch_size, kd_batch_size, num_clients, synthetic_path)e
            dataset.prepare_data(data_partition)

            client_models = ClientModelStrategy.available[models](num_clients)

            for i in range(len(client_models)):
                torch.cuda.empty_cache()
                torch.manual_seed(i)
                client = client_models[i]()
                client = client(1,10)
                client.to('cuda')

                opt = optim.SGD(client.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(25):
                    client.train()
                    total_loss = 0
                    total_correct = 0
                    total = 0
                    for idx, (data, target) in enumerate(dataset.client_dataloaders[i]):
                        data, target = data.to('cuda'), target.to('cuda')
                        opt.zero_grad()
                        output = client(data)
                        loss = criterion(output, target)
                        loss.backward()
                        opt.step()
                        total += target.size(0)
                        total_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        total_correct += (predicted == target).sum().item()

                    logger.add_scalar(f'Training_Loss/Client_{i:02}', total_loss/len(dataset.client_dataloaders[i]), epoch)
                    logger.add_scalar(f'Training_Accuracy/Client_{i:02}', 100*total_correct/total, epoch)

                    if epoch % 5 == 0:
                        client.eval()
                        with torch.no_grad():
                            test_total_loss = 0
                            test_total_correct = 0
                            test_total = 0
                            for idx, (data, target) in enumerate(dataset.test_dataloader):
                                data, target = data.to('cuda'), target.to('cuda')
                                output = client(data)
                                loss = criterion(output, target)

                                test_total += target.size(0)
                                test_total_loss += loss.item()
                                _, predicted = torch.max(output.data, 1)
                                test_total_correct += (predicted == target).sum().item()

                            logger.add_scalar(f'Validation_Loss/Client_{i:02}', test_total_loss/len(dataset.test_dataloader), epoch)
                            logger.add_scalar(f'Validation_Accuracy/Client_{i:02}', 100*test_total_correct/test_total, epoch)
                    logger.flush()

                torch.save({
                        'model_state_dict': client.state_dict(),
                        'optimizer_state_dict': opt.state_dict()
                        },f"client_{i:02}.pt")
                                