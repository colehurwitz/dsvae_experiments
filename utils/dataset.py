import torch
from torch.utils.data import Dataset, DataLoader


class DSVAE_DATA(Dataset):
    """DSVAE dataset."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    
class DSVAE_DATA_HR(Dataset):
    """DSVAE HR dataset."""

    def __init__(self, x, x_hr, y):
        self.x = x
        self.x_hr = x_hr
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        x_hr = self.x_hr[idx]
        y = self.y[idx]
        return x, x_hr, y

class VAE_DATA(Dataset):
    """DSVAE HR dataset."""

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        return x