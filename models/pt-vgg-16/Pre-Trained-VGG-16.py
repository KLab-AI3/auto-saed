import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import vgg16

import numpy as np

class TEMDataset(Dataset):
    def __init__(self, csv_key):
        self.csv_key = pd.read_csv(csv_key, header = None)
    
    def __len__(self):
        return len(self.csv_key)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = self.csv_key.iloc[index, 3]

        image_1 = io.imread(image_path, as_gray = True)
        image_2 = io.imread(image_path, as_gray = True)
        image_3 = io.imread(image_path, as_gray = True)
        
        image = np.zeros((3, image_1.shape[0], image_1.shape[1]))
        image[0, :, :] = image_1
        image[1, :, :] = image_2
        image[2, :, :] = image_3

        sample = {'image': torch.tensor(image).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_train = TEMDataset(r'./../../testing/no_aug_data/new_24_compha_train_key.csv')

training_loader = DataLoader(PatternData_train, batch_size = 16, shuffle = True, num_workers = 8)

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

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor(images).flatten(1)
        x = self.classifier(x)
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
       
    def configure_optimizers(self):
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001)
        return optimizer

PatternNetVGG16 = TEMVGG16()

print(PatternNetVGG16)

for i in range(5, 10):
    PatternNetVGG16 = TEMVGG16()

    old_layer_before = PatternNetVGG16.feature_extractor[0][0].state_dict()['weight'].numpy()
    new_layer_before = PatternNetVGG16.classifier[0].state_dict()['weight'].numpy()

    trainer = pl.Trainer(max_epochs = 20, devices = -1, accelerator = 'gpu', strategy = 'ddp_find_unused_parameters_true')
    trainer.fit(model = PatternNetVGG16, train_dataloaders = training_loader)

    old_layer_after = PatternNetVGG16.feature_extractor[0][0].state_dict()['weight'].numpy()
    new_layer_after = PatternNetVGG16.classifier[0].state_dict()['weight'].numpy()

    print("The first old layer's weights have not changed during training:", np.array_equal(old_layer_before, old_layer_after))
    print("The new layer's weights have not changed during training:", np.array_equal(new_layer_before, new_layer_after))

    torch.save(PatternNetVGG16.state_dict(), f'./state-dicts/Pre-Trained-VGG-16-{i}.pt')