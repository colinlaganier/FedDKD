import torch
from federated.Dataset import Dataset
from models.Models import Models
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

model = Models.available['resnet18']()(num_classes=10).to('cuda')
dataset = Dataset(data_path="dataset\\cinic-10", dataset_id="cinic10", batch_size=128, kd_batch_size=128, num_clients=5, synthetic_path=None)
dataset.prepare_data("iid")
dataloader = dataset.client_dataloaders[0]
testloader = dataset.test_dataloader
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(150):
    total_loss = 0
    total_correct = 0
    total = 0
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        model.train()
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total += target.size(0)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(dataloader.dataset),
        #     100. * batch_idx / len(dataloader), loss.item()))
    writer.add_scalar('Loss/train', total_loss/len(dataloader), epoch)
    writer.add_scalar('Accuracy/train', 100*total_correct/total, epoch)
    model.eval()    
    with torch.no_grad():
        test_total_loss = 0
        test_total_correct = 0
        test_total = 0
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            loss = criterion(output, target)

            test_total += target.size(0)
            test_total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_total_correct += (predicted == target).sum().item()

        # Log statistics
        writer.add_scalar('Loss/test', test_total_loss/len(testloader), epoch)
        writer.add_scalar('Accuracy/test', 100*test_total_correct/test_total, epoch)
        writer.flush()
