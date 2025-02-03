import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import vgg16

from torchmetrics import Accuracy

import csv

import numpy as np

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
        
        tri_layer_image = np.zeros((3, img.shape[0], img.shape[1]))
        tri_layer_image[0, :, :] = img
        tri_layer_image[1, :, :] = img
        tri_layer_image[2, :, :] = img

        sample = {'image': torch.tensor(tri_layer_image).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_test = TEMTESTDataset(r'./../../testing/no_aug_data/new_compha_test_key.csv')

testing_loader = DataLoader(PatternData_test, batch_size = 16, shuffle = False, num_workers = 8)

class TEMVGG16(pl.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = vgg16(weights = 'DEFAULT')
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor[1] = nn.AdaptiveAvgPool2d(output_size=(25, 25))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=320000, out_features=4096, bias=True), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5, inplace=False), 
            nn.Linear(in_features=4096, out_features=4096, bias=True), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5, inplace=False), 
            nn.Linear(in_features=4096, out_features=9, bias=True)
        )
        
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 9)
        self.set_check = []
        self.preder = []
        self.labeler = []
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.feature_extractor(images).flatten(1)
        x = self.classifier(x)
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

PatternNetVGG16 = TEMVGG16()

print(PatternNetVGG16)

for i in range(5, 10):
    PatternNetVGG16 = TEMVGG16()
    PatternNetVGG16.load_state_dict(torch.load(f'./state-dicts/Pre-Trained-VGG-16-{i}.pt'))
    PatternNetVGG16.eval()

    tester = pl.Trainer(devices = 1, accelerator = 'gpu', num_nodes = 1)
    tester.test(model = PatternNetVGG16, dataloaders = testing_loader)

    print("length", len(PatternNetVGG16.set_check))
    print("correct", sum(bool(x) for x in PatternNetVGG16.set_check))
    print("percent", sum(bool(x) for x in PatternNetVGG16.set_check) / len(PatternNetVGG16.set_check))

    with open(f'./preds/Pre-Trained-VGG-16-{i}.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Prediction', 'Correct'])
        writer.writerows(zip(PatternNetVGG16.labeler, PatternNetVGG16.preder, PatternNetVGG16.set_check))