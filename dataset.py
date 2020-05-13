import os
from torch.utils.data import Dataset
from PIL import Image

# Imagenet Dataset that replicates the ImageFolder implementation
class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)
        self.transform = transform
        
        print(self.root_dir)
        print(self.img_list)
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_list[idx]))
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img