import argparse
import os
import json
import time
import torch
from models.Models import Models
from models.ClientModelStrategy import ClientModelStrategy
from federated import Client, Dataset, Server, Scheduler
from torch.utils.tensorboard import SummaryWriter

def main(args, logger):

    num_devices = torch.cuda.device_count()
    print("Number of GPUs available: {}".format(num_devices))

    # Create dataset object splitting data among clients
    dataset = Dataset(args.data_path, args.dataset_id, args.image_size, args.batch_size, args.num_clients)
    dataset.prepare_data()

    # Create client model strategy
    if args.client_model in Models.available:
        client_models = ClientModelStrategy.homogenous(args.num_clients, Models.available[args.client_model])
    else:
        client_models = ClientModelStrategy.available[args.client_model](args.num_clients)

    # # Create clients and assign data and models
    # clients = [None] * args.num_clients
    # for client_id in range(args.num_clients):
    #     clients[client_id] = Client(client_id)
    #     clients[client_id].set_data(dataset.client_dataloaders[client_id])
    #     clients[client_id].set_model(client_models[client_id])

    #     if args.load_diffusion:
    #         clients[client_id].get_diffusion()

    # Create server
    # server = Server(arg./s.server_model)

    # Create scheduler
    scheduler = Scheduler(args.num_clients, num_devices, args.server_model, client_models, dataset, args.epochs, args.batch_size, args.load_diffusion, logger)

    scheduler.init_training()

    # scheduler.init_diffusion(args.load_diffusion)
    # scheduler.init_clients()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--server-model", type=str, choices=list(Models.available.keys()), default="resnet32")
    parser.add_argument("--client-model", type=str, choices=list(Models.available.keys()) + list(ClientModelStrategy.available.keys()), default="strategy_1")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--load-diffusion", type=bool, default=False)
    parser.add_argument("--save-checkpoint", type=bool, default=False)
    
    # print(list(Models.available.keys()))

    # parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    # parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # parser.add_argument("--num-workers", type=int, default=4)
    # parser.add_argument("--log-every", type=int, default=100)
    # parser.add_argument("--ckpt-every", type=int, default=50_000)
    # parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() >= args.num_clients, "Not enough GPUs available" #change
    torch.backends.cudnn.benchmark = False

    # Create tensorboard writer
    logger = SummaryWriter()

    # Create checkpoint directory for run
    if (args.save_checkpoint):
        # Create checkpoint directory
        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"checkpoints/{timestr}", exist_ok=True)
        # Save command line arguments
        with open(f"checkpoints/{timestr}/commandline_args.txt", 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    
    main(args, logger)