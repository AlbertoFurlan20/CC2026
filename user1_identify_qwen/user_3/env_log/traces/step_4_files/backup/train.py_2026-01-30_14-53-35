import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
from encode import rle_encode, list_to_string
from false_color import false_color  # Assuming false_color is defined in a separate file

def dice_score(y_p, y_t, smooth=1e-6):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_p = F.softmax(y_p, dim=1)
    y_p = torch.argmax(y_p, dim=1, keepdim=True)
    i = torch.sum(y_p * y_t, dim=(2, 3))
    u = torch.sum(y_p, dim=(2, 3)) + torch.sum(y_t, dim=(2, 3))
    score = (2 * i + smooth) / (u + smooth)
    return torch.mean(score)

def ce_loss(y_p, y_t):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_t = y_t.squeeze(dim=1)
    weight = torch.Tensor([0.57, 4.17]).to(y_t.device)
    criterion = nn.CrossEntropyLoss(weight)
    loss = criterion(y_p, y_t)
    return loss

class ICRGWDataset(Dataset):
    def __init__(self, tar_path, ids, padding_size):
        self.tar_path = tar_path
        self.ids = ids
        self.padding_size = padding_size
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        N_TIMES_BEFORE = 4
        sample_path = f"{self.tar_path}/{self.ids[idx]}"
        band11 = np.load(f"{sample_path}/band_11.npy")[..., N_TIMES_BEFORE]
        band14 = np.load(f"{sample_path}/band_14.npy")[..., N_TIMES_BEFORE]
        band15 = np.load(f"{sample_path}/band_15.npy")[..., N_TIMES_BEFORE]
        image = false_color(band11, band14, band15)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        padding_size = self.padding_size
        image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
        
        try:
            label = np.load(f"{sample_path}/human_pixel_masks.npy")
            label = torch.Tensor(label).to(torch.int64)
            label = label.permute(2, 0, 1)
        except FileNotFoundError:
            # label does not exist
            label = torch.zeros((1, image.shape[1], image.shape[2]))
        
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(data_path, epochs=10, batch_size=8, padding_size=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_ids, val_ids = train_test_split(os.listdir(data_path), test_size=0.2, random_state=42)
    train_dataset = ICRGWDataset(data_path, train_ids, padding_size)
    val_dataset = ICRGWDataset(data_path, val_ids, padding_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_val_dice =