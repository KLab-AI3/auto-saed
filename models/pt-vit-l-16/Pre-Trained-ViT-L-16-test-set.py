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
        
        # Resizing to 224 x 224 to maintain the same size used in the transfer model.
        img = resize(img, (224, 224), anti_aliasing = True)
        
        image = np.zeros((3, img.shape[0], img.shape[1]))
        image[0, :, :] = img
        image[1, :, :] = img
        image[2, :, :] = img

        sample = {'image': torch.tensor(image).float(), 'label': torch.tensor(self.csv_key.iloc[index, 1])}

        return sample

PatternData_test = TEMTESTDataset(r'./../../testing/no_aug_data/new_compha_test_key.csv')

testing_loader = DataLoader(PatternData_test, batch_size = 16, shuffle = False, num_workers = 8)

class TEMVITL16(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vit_l_16(weights = 'DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.heads.head = nn.Linear(in_features=1024, out_features=9, bias=True)
        
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 9)
        self.set_check = []
        self.preder = []
        self.labeler = []
    
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
       
    def configure_optimizers(self):
        # Beta is the default and weight_decay is that listed in the paper: https://arxiv.org/pdf/2010.11929.pdf
        # Though, they are using a much larger batch size of 4,096.
        optimizer = optim.Adam(params = self.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.1)
        return optimizer

PatternNetVITL16 = TEMVITL16()

print(PatternNetVITL16)

for i in range(10):
    PatternNetVITL16 = TEMVITL16()
    PatternNetVITL16.load_state_dict(torch.load(f'./state-dicts/Pre-Trained-ViT-L-16-{i}.pt'))
    PatternNetVITL16.eval()

    tester = pl.Trainer(devices = 1, accelerator = 'gpu', num_nodes = 1)
    tester.test(model = PatternNetVITL16, dataloaders = testing_loader)

    print("length", len(PatternNetVITL16.set_check))
    print("correct", sum(bool(x) for x in PatternNetVITL16.set_check))
    print("percent", sum(bool(x) for x in PatternNetVITL16.set_check) / len(PatternNetVITL16.set_check))

    with open(f'./preds/Pre-Trained-ViT-L-16-{i}.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Prediction', 'Correct'])
        writer.writerows(zip(PatternNetVITL16.labeler, PatternNetVITL16.preder, PatternNetVITL16.set_check))