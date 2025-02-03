import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import efficientnet_b3

from torchmetrics import Accuracy

import csv

class TEMDataset(Dataset):
    def __init__(self, csv_key):
        self.csv_key = pd.read_csv(csv_key, header = None)
    
    def __len__(self):
        return len(self.csv_key)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.csv_key.iloc[index, 3]

        image = io.imread(image_path, as_gray = True)

        sample = {'image': torch.tensor(image).unsqueeze(0).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

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

PatternData_train = TEMDataset(r'./../../testing/no_aug_data/new_24_compha_train_key.csv')
PatternData_test = TEMTESTDataset(r'./../../testing/no_aug_data/new_compha_test_key.csv')

training_loader = DataLoader(PatternData_train, batch_size = 32, shuffle = True, num_workers = 8)
testing_loader = DataLoader(PatternData_test, batch_size = 16, shuffle = False, num_workers = 8)

class TEMENB3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b3(weights = None)
        self.model.features[0][0] = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=9, bias=True)
        
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 9)
        self.set_check = []
        self.preder = []
        self.labeler = []
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.model(images)
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.model(images)
        preds = x.argmax(dim = -1)
        sc = [i == j for i, j in zip(preds.cpu().numpy(), labels.cpu().numpy())]
        self.set_check += sc
        self.preder += preds.cpu()
        self.labeler += labels.cpu()
        self.accuracy(x, labels)
        self.log("test_accuracy", self.accuracy, on_epoch = True)
    
#     Using adam here, original paper uses RMSprop
    def configure_optimizers(self):
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001)
        return optimizer

#     def configure_optimizers(self):
#             optimizer = optim.RMSprop(params = self.parameters(), lr = 0.256, momentum = 0.9, 
#                                      weight_decay = 1e-5)
#             return optimizer

PatternNetENB3 = TEMENB3()

print(PatternNetENB3)

for i in range(10):
    PatternNetENB3 = TEMENB3()
    PatternNetENB3.load_state_dict(torch.load(f'./state-dicts/Raw-EfficientNet-B3-{i}.pt'))
    PatternNetENB3.eval()

    tester = pl.Trainer(devices = 1, accelerator = 'gpu', num_nodes = 1)
    tester.test(model = PatternNetENB3, dataloaders = testing_loader)

    print("length", len(PatternNetENB3.set_check))
    print("correct", sum(bool(x) for x in PatternNetENB3.set_check))
    print("percent", sum(bool(x) for x in PatternNetENB3.set_check) / len(PatternNetENB3.set_check))
    
    with open(f'./preds/Raw-Predictions-EfficientNet-B3-{i}.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Prediction', 'Correct'])
        writer.writerows(zip(PatternNetENB3.labeler, PatternNetENB3.preder, PatternNetENB3.set_check))