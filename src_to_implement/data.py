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
        if flag == 'train':
            self.__transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((300, 300)),
                tv.transforms.RandomHorizontalFlip(),       # data augmentation for training
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        elif flag == 'val':
            self.__transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((300, 300)),       #(300,300) for running tests
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = imread(self.data.iloc[idx, 0])
        img = gray2rgb(img)
        img = self.__transform(img)
        crack_label = self.data.iloc[idx, 1]  
        inactive_label = self.data.iloc[idx, 2]  
        labels = torch.tensor([crack_label, inactive_label], dtype=torch.float32)
        return img, labels
