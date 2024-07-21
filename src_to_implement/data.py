from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # flag can only either be 'val' or 'train'
    def __init__(self, data, flag='train'):
        self.data = data
        self.flag = flag
        self.__transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = imread(self.data.iloc[idx, 0])
        img = gray2rgb(img)
        img = self.__transform(img)
        if self.flag == 'train':
            return img, torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        else:
            return img
