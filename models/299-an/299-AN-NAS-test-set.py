import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

from torchmetrics import Accuracy

import pytorch_lightning as pl

import numpy as np

import csv

# DDP seems to just run the script twice --> this appears two times. 
# I think it just splits up the training dataset into two different 
# groupings (one for each GPU) and automates the syncing with sync_dist.
print("The number of CUDA-enabled devices is:", torch.cuda.device_count())
    
class TEMTESTDataset(Dataset):
    def __init__(self, csv_key):
        self.csv_key = pd.read_csv(csv_key, header = None)
    
    def __len__(self):
        return len(self.csv_key)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.csv_key.iloc[index, 3]

        image = io.imread(image_path, as_gray = True)
        
        height, width = image.shape
        
        new_width = new_height = 800
        
        left = int((width - new_width)/2)
        top = int((height - new_height)/2)
        right = int((width + new_width)/2)
        bottom = int((height + new_height)/2)
        
        img = image[left:right, top:bottom]

        sample = {'image': torch.tensor(img).unsqueeze(0).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_test = TEMTESTDataset(r'./../../testing/no_aug_data/new_compha_test_key.csv')

testing_loader = DataLoader(PatternData_test, batch_size = 8, shuffle = False, num_workers = 8)

# # No same padding.
class TEMCNN(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TEMCNN, self).__init__()
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32), 

            nn.Conv2d(in_channels = 32, out_channels = 32, 
                        kernel_size = (3, 3), stride = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2), 
            nn.BatchNorm2d(num_features = 32)
        )

        # Get input size.
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = 3200, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64), 

            nn.Linear(in_features = 64, out_features = 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features = 64),
            
            nn.Linear(in_features = 64, out_features = 9), 
            nn.Sigmoid()
        )
        
        # Define accuracy in the __init__ to have it on the same device as the input.
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 9)
        self.set_check = []
        self.preder = []
        self.labeler = []
    
    # Can always add in a validation step as validation_step().
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.convolution_layers(images)
        x = torch.flatten(input = x, start_dim = 1)
        x = self.linear_layers(x)
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.convolution_layers(images)
        x = torch.flatten(input = x, start_dim = 1)
        x = self.linear_layers(x)
        preds = x.argmax(dim = -1)
        sc = [i == j for i, j in zip(preds.cpu().numpy(), labels.cpu().numpy())]
        self.set_check += sc
        self.preder += preds.cpu()
        self.labeler += labels.cpu()
        self.accuracy(x, labels)
        self.log("test_accuracy", self.accuracy, on_epoch = True)

       
    def configure_optimizers(self):
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001)
        return optimizer

PatternNet = TEMCNN()

print(PatternNet)

for i in range(10):
    PatternNet = TEMCNN()
    PatternNet.load_state_dict(torch.load(f'./state-dicts/299-AN-NAS-{i}.pt'))
    PatternNet.eval()

    # Can't run these in the same script, can try ddp_spawn --> says that it has its own limitations.
    tester = pl.Trainer(devices = 1, accelerator = 'gpu', num_nodes = 1)
    tester.test(model = PatternNet, dataloaders = testing_loader)

    print("length", len(PatternNet.set_check))
    print("correct", sum(bool(x) for x in PatternNet.set_check))
    print("percent", sum(bool(x) for x in PatternNet.set_check) / len(PatternNet.set_check))
    
    with open(f'./preds/Predictions-299-AN-NAS-{i}.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Prediction', 'Correct'])
        writer.writerows(zip(PatternNet.labeler, PatternNet.preder, PatternNet.set_check))