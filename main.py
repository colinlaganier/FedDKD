import argparse
import pathlib
import os
import json
import signal
import time
import torch
import torch.multiprocessing as mp
from models.Models import Models
from models.ClientModelStrategy import ClientModelStrategy
from federated.Scheduler import Scheduler
from torch.utils.tensorboard import SummaryWriter
from functools import partial

def main(args, checkpoint_path, logger):
    """
    Main function for running the federated learning process

    Args:
        args (argparse.Namespace): command line arguments
        checkpoint_path (str): path to save checkpoints
        logger (SummaryWriter): tensorboard logger
    """ 
    global scheduler

    num_devices = torch.cuda.device_count()
    print("Number of GPUs available: {}".format(num_devices))

    # Create client model strategy
    if args.client_model in Models.available:
        client_models = ClientModelStrategy.homogenous(args.num_clients, Models.available[args.client_model])
    else:
        client_models = ClientModelStrategy.available[args.client_model](args.num_clients)

    # Create scheduler
    scheduler = Scheduler(num_devices,
                          args.num_clients,  
                          args.server_model, 
                          client_models, 
                          args.epochs, 
                          args.kd_epochs, 
                          args.batch_size, 
                          args.kd_batch_size, 
                          args.dataset_path,
                          args.dataset_id, 
                          args.data_partition,
                          args.synthetic_path, 
                          args.load_diffusion, 
                          args.save_checkpoint,
                          checkpoint_path,
                          logger)

    scheduler.train(args.num_rounds)
    # scheduler.train_baseline()

def handler(save_checkpoint, signum, frame):
    """
    Handler for terminating the program with Ctrl-C

    Args:
        signum (int): signal number
        frame (frame): current stack frame
    """
    if save_checkpoint:
        res = input("Ctrl-C was pressed. Do you want to save client and server checkpoints y/n ")
        if res == 'y':
            scheduler.save_checkpoints()
            print("Saved client and server checkpoints")
    exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, required=True, choices=["cinic10", "emnist"], default="emnist")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-partition", type=str, choices=["iid", "dirichlet", "random"], default="iid")
    parser.add_argument("--synthetic-path", type=str, default=None)
    parser.add_argument("--server-model", type=str, choices=list(Models.available.keys()), default="resnet34")
    parser.add_argument("--client-model", type=str, choices=list(Models.available.keys()) + list(ClientModelStrategy.available.keys()), default="strategy_1")
    # parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--kd-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kd-batch-size", type=int, default=32)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--load-diffusion", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if (args.dataset_id == "cinic10"):
        assert args.dataset_path is not None, "Dataset path must be specified for cinic10"

    # Register handler for termination
    signal.signal(signal.SIGINT, partial(handler, args.save_checkpoint))

    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.backends.cudnn.benchmark = False

    # Create tensorboard writer
    logger = SummaryWriter()

    # Create checkpoint directory for run
    if (args.save_checkpoint):
        # Create checkpoint directory
        checkpoint_path = "checkpoints/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"{checkpoint_path}", exist_ok=True)
        # Save command line arguments
        with open(f"{checkpoint_path}/commandline_args.txt", 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        checkpoint_path = None
    
    main(args, checkpoint_path, logger)