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

        image = io.imread(image_path, as_gray = True)
        
        # Resizing to 224 x 224 to maintain the same size used in the transfer model.
        img = resize(image, (224, 224), anti_aliasing = True)
        
        image = np.zeros((3, img.shape[0], img.shape[1]))
        image[0, :, :] = img
        image[1, :, :] = img
        image[2, :, :] = img

        sample = {'image': torch.tensor(image).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_train = TEMDataset(r'./../../testing/no_aug_data/new_24_compha_train_key.csv')

training_loader = DataLoader(PatternData_train, batch_size = 16, shuffle = True, num_workers = 8)

class TEMVITL16(pl.LightningModule):
    def __init__(self):
        super().__init__()
#         backbone = vit_l_16(weights = 'DEFAULT')
#         layers = list(backbone.children())[:-1]
#         self.feature_extractor = nn.Sequential(*layers)
#         self.classifier = nn.Linear(in_features=1024, out_features=9, bias=True)
        self.model = vit_l_16(weights = 'DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
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

    old_layer_before = PatternNetVITL16.model.encoder.layers.encoder_layer_0.mlp[0].state_dict()['weight'].numpy()
    new_layer_before = PatternNetVITL16.model.heads.head.state_dict()['weight'].numpy()

    trainer = pl.Trainer(max_epochs = 20, devices = -1, accelerator = 'gpu', strategy = 'ddp')
    trainer.fit(model = PatternNetVITL16, train_dataloaders = training_loader)

    old_layer_after = PatternNetVITL16.model.encoder.layers.encoder_layer_0.mlp[0].state_dict()['weight'].numpy()
    new_layer_after = PatternNetVITL16.model.heads.head.state_dict()['weight'].numpy()

    print("The first old layer's weights have not changed during training:", np.array_equal(old_layer_before, old_layer_after))
    print("The new layer's weights have not changed during training:", np.array_equal(new_layer_before, new_layer_after))

    torch.save(PatternNetVITL16.state_dict(), f'./state-dicts/Pre-Trained-ViT-L-16-{i}.pt')