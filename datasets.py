import os
from PIL import Image
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Vanilla Imagenet dataset that replicates ImageFolder
class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_list[idx]))
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img


class SampleDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.data = np.load(file_path)['x']
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])

        if self.transform:
            img = self.transform(img)
        
        return img