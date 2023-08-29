import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from models.ClientModelStrategy import ClientModelStrategy
from models.Models import Models
from knowledge_distillation import SoftTarget

def cutmix(data, targets, alpha=0.25):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    # shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)

    return data, None

def adjust_learning_rate(epoch, learning_rate, lr_decay_rate, lr_decay_epochs, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def main():
    logger = SummaryWriter()
    strategy = 'strategy_1'
    num_clients = 5
    num_epochs = 250
    learning_rate = 0.05
    lr_decay_rate = 0.1
    lr_decay_epochs = [150,180,210]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client_models = ClientModelStrategy.available[strategy](num_clients)
    for i in range(num_clients):
        # init models
        client_models[i] = client_models[i]()(10)
        checkpoint = torch.load('checkpoints/pretrain/client_0{}.pt'.format(i))
        client_models[i].load_state_dict(checkpoint['model_state_dict'])
        client_models[i].to(device)

    server = Models.ResNet18()(10)
    server.to(device)

    train_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.ImageFolder(root='../FederatedDiffusionModels/DDPM/synthetic/5K', transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2)

    test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    # testset = torchvision.datasets.ImageFolder(root='data/synthetic', transform=test_transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # criterion = nn.CrossEntropyLoss()
    kd_criterion = SoftTarget(10)

    optimizer = optim.SGD(server.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for model in client_models:
        model.eval()
    server.train()
    logits = [] 
    # [[]]*num_clients


    for epoch in range(num_epochs):  # loop over the dataset multiple times

        adjust_learning_rate(epoch, learning_rate, lr_decay_rate, lr_decay_epochs, optimizer)

        total_loss = 0
        total_correct = 0
        total = 0
        total_loss = 0

        for i, (inputs, labels) in enumerate(trainloader, 0):
                
                inputs, _ = cutmix(inputs, labels)

                with torch.no_grad():

                    for model in client_models:     

                        inputs = inputs.to(device)

                        preds = model(inputs)
                        logits.append(preds)

                    teacher_logits = torch.mean(torch.stack(logits), dim=0)
        
                optimizer.zero_grad()

                preds = server(inputs)
                # loss = criterion(preds, labels)

                loss = kd_criterion(preds, teacher_logits)
                # loss += kd_loss

                loss.backward()
                optimizer.step()

                logits.clear()
                # for client_logits in logits:
                    # client_logits.clear()

        if epoch % 5 == 0:

            for i, (inputs, labels) in enumerate(testloader):

                inputs = inputs.to(device)

                preds = server(inputs)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                total_correct += (predicted.cpu() == labels).sum().item()

            # logger.add_scalar('Loss/test', loss.item(), epoch)
            logger.add_scalar('Accuracy/test', 100*total_correct/total, epoch)

if __name__ == "__main__":
    main()