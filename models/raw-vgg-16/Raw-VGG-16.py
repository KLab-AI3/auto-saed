import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import vgg16

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

PatternData_train = TEMDataset(r'./../../testing/no_aug_data/new_24_compha_train_key.csv')

training_loader = DataLoader(PatternData_train, batch_size = 16, shuffle = True, num_workers = 8)

class TEMVGG16(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = vgg16(weights = None)
        self.features = model.features
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(25, 25)) # was 7 x 7
        self.classifier = model.classifier
        self.classifier[0] = nn.Linear(in_features=320000, out_features=4096, bias=True)
        self.classifier[6] = nn.Linear(in_features = 4096, out_features = 9, bias = True)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.features(images)
#         print(x.shape)
#         x = torch.flatten(input = x, start_dim = 1)
        x = self.avgpool(x).flatten(1) # necessary???
#         print(x.shape)
        x = self.classifier(x)
        #x = x.Softmax() --> can add in for inference?
        #https://github.com/pytorch/vision/issues/432
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
       
    def configure_optimizers(self):
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001)
        return optimizer

PatternNetVGG16 = TEMVGG16()

print(PatternNetVGG16)

for i in range(7, 10):
    PatternNetVGG16 = TEMVGG16()
    
    trainer = pl.Trainer(max_epochs = 20, devices = -1, accelerator = 'gpu', strategy = 'ddp')
    trainer.fit(model = PatternNetVGG16, train_dataloaders = training_loader)

    torch.save(PatternNetVGG16.state_dict(), f'./state-dicts/Raw-VGG-16-{i}.pt')