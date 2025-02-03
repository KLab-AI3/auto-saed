import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io
from skimage.transform import resize

import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from torchvision.models import vit_l_16

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
        
        # Resizing to 224 x 224 to maintain the same size used in the transfer model.
        image = resize(image, (224, 224), anti_aliasing = True)

        sample = {'image': torch.tensor(image).unsqueeze(0).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_train = TEMDataset(r'./../../testing/no_aug_data/new_24_compha_train_key.csv')

training_loader = DataLoader(PatternData_train, batch_size = 16, shuffle = True, num_workers = 8)

class TEMVITL16(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vit_l_16(weights = None)
        self.model.conv_proj = nn.Conv2d(1, 1024, kernel_size=(16, 16), stride=(16, 16))
        self.model.heads.head = nn.Linear(in_features=1024, out_features=9, bias=True)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        x = self.model(images)
        loss = nn.CrossEntropyLoss()
        train_loss = loss(x, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = True, prog_bar = True, logger = False, sync_dist = True)
        return train_loss
       
    def configure_optimizers(self):
        # Beta is the default and weight_decay is that listed in the paper: https://arxiv.org/pdf/2010.11929.pdf
        # Though, they are using a much larger batch size of 4,096.
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.1)
        return optimizer

PatternNetVITL16 = TEMVITL16()

print(PatternNetVITL16)

for i in range(10):
    PatternNetVITL16 = TEMVITL16()
    
    trainer = pl.Trainer(max_epochs = 20, devices = -1, accelerator = 'gpu', strategy = 'ddp')
    trainer.fit(model = PatternNetVITL16, train_dataloaders = training_loader)

    torch.save(PatternNetVITL16.state_dict(), f'./state-dicts/Raw-ViT-L-16-{i}.pt')