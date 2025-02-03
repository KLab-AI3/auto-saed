import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import efficientnet_b3

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

class TEMENB3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b3(weights = None)                            
        self.model.features[0][0] = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=9, bias=True)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.model(images)
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
    
    # Using adam here, original paper uses RMSprop
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
    
    trainer = pl.Trainer(max_epochs = 20, devices = -1, accelerator = 'gpu', strategy = 'ddp')
    trainer.fit(model = PatternNetENB3, train_dataloaders = training_loader)

    torch.save(PatternNetENB3.state_dict(), f'./state-dicts/Raw-EfficientNet-B3-{i}.pt')