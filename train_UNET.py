import os, sys
import torch
import torchvision.transforms as transforms
import wandb

from arguments import parse_args
from dataset import ImagenetDataset
from models import VQVAE
from train import train

import sys
import os
sys.path.insert(0, 'Pytorch-UNet')
sys.path.insert(0, 'utils')
sys.path.insert(0, 'models')
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import plot_pytorch_images, make_grid
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision
from misc import merge
from unet.unet_model import UNet
from torch.nn import functional as F
from logger import Logger


if __name__ == "__main__":
    # Set up logger
    logger = Logger()

    # Parse arguments & log
    args = parse_args()
    logger.update_args(args)
    
    # Accelerate training with benchmark true
    torch.backends.cudnn.benchmark = True

    # Create output directory
    try:
        os.mkdir(args.output_dir)
    except:
        raise Exception('Cannot create output directory')

    # Initialize wandb
    wandb.init(project=args.project)
    
    # Create datasets
    default_transform = transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.CenterCrop(args.image_size),
                            transforms.ToTensor()
                        ])

    # Create training dataset
    
    train_dataset = dset.ImageFolder(root=args.train_dir, transform=default_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    # Create validation dataset
    
    valid_data = dset.ImageFolder(root=args.valid_dir, transform=default_transform)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)
    
    model = UNet(n_channels=3, n_classes=3, bilinear=True)
    model.to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0)

    state_dict = {'itr': 0}
    
    for epoch in range(args.num_epochs):
        train(epoch, state_dict, model, optimizer, train_loader, valid_loader, args, logger)