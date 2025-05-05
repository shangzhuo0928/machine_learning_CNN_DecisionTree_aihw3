import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

n1 = 32
n2, n3 = n1*2, n1*4
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # Design your CNN, it can be no more than 3 convolution layers
        super(CNN, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=5) #224-4 220/2
        self.Conv_2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=5) #110-4 106/2
        self.Conv_3 = nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=5) #53-4 49/2 = 24
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(24*24*n3, 128) #units=128
        self.fc_2 = nn.Linear(128, num_classes) #output [0 1 2 3 4] len=5
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Forward the model
        x = self.Conv_1(x)
        x = torch.relu(x)
        x = self.Maxpool(x) #layer 1

        x = self.Conv_2(x)
        x = torch.relu(x)
        x = self.Maxpool(x) #layer 2

        x = self.Conv_3(x)
        x = torch.relu(x)
        x = self.Maxpool(x) #layer 3

        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = torch.relu(x) #fc1

        x = self.dropout(x)
        x = self.fc_2(x) #fc2
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()

    all_loss = 0.0
    batch = len(train_loader)

    with tqdm(total=batch, desc="Training Progress", unit="batch") as pbar:
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            all_loss += loss.item()
            
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
    
    avg_loss = all_loss / batch

    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()

    all_loss = 0.0
    correct = 0
    batch = len(val_loader)
    num = 0
    with tqdm(total=batch, desc="Validating Progress", unit="batch") as pbar:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                num += len(images) 
                outputs = model(images)
    
                loss = criterion(outputs, labels)
                all_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
    
    avg_loss = all_loss / batch
    accuracy = 100*correct / num

    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    data = [] 
    dataset = test_loader.dataset
    batch = len(test_loader)

    with tqdm(total=batch, desc="Testing Progress", unit="batch") as pbar:
        with torch.no_grad():
            for batch_idx, (images, l) in enumerate(test_loader):
                images = images.to(device)

                output = model(images)
                _, predicted = torch.max(output, 1)
                for i in range(len(images)):
                    idx = batch_idx*32 + i
                    path = dataset.image[idx]
                    path = path.replace('data/test/', '', 1)
                    path = path.replace('.jpg', '', 1)
                    data.append((path, predicted[i].item()))
                
                pbar.update(1)
    
    df = pd.DataFrame(data, columns=['id', 'prediction'])
    df.to_csv('CNN.csv', index=False)

    print(f"Predictions saved to 'CNN.csv'")
    return