import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import wandb
from UNET_utils import *

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

loss_func = torch.nn.MSELoss(reduction='sum')
def train(epoch, model, optimizer, train_loader, val_loader, args, LOGGER):
    model.train()
    
    # Train loop
    train_loss = 0
    num_examples = 0
    for i, data in enumerate(train_loader):
        x = data[0]
        y = F.interpolate(F.interpolate(x, args.low_resolution, mode="bilinear"), args.image_size, mode="bilinear")
        optimizer.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)
        x_mask = x - y 
        x_mask_hat = model(y)
        x_hat = y + x_mask_hat
        loss = loss_func(x_mask_hat, x_mask)
        train_loss += loss.item()
        num_examples += x_hat.shape[0]
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("{}".format(i))
            wandb.log({'train_loss':train_loss/num_examples})
            
    
    # Val loop
    model.eval()
    val_loss = 0
    num_examples = 0
    for i, data in enumerate(val_loader):
        x_val = data[0]
        y_val = F.interpolate(F.interpolate(x_val, args.low_resolution, mode="bilinear"), args.image_size, mode="bilinear")
        x_val = x_val.to(args.device)
        y_val = y_val.to(args.device)    
        x_mask_val = x_val - y_val
        x_mask_hat_val = model(y_val)
        x_hat_val = y_val + x_mask_hat_val
        loss_val = loss_func(x_mask_hat_val, x_mask_val)
        val_loss += loss_val.item()
        num_examples += x_hat_val.shape[0]
        
    avg_val_loss = val_loss/num_examples
    wandb.log({'val_loss':avg_val_loss})
    
    # Train image logging and saving
    fig = make_grid(x_mask[:5], num_images=5, title='training mask, num epochs: {}'.format(epoch))
    wandb.log({'training mask':fig})   
    save_image(x_mask[:5].cpu(), args.output_dir + 'training_mask{}.png'.format(epoch))
    
    fig = make_grid(x_mask_hat[:5], num_images=5, title='UNET train mask reconstruction, num epochs: {}'.format(epoch))
    wandb.log({'UNET train mask reconstruction':fig})
    save_image(x_mask_hat[:5].cpu(), args.output_dir + 'UNET_train_mask_reconstruction{}.png'.format(epoch))
    
    fig = make_grid(y[:5], num_images=5, title='training y {} -> {}, num epochs: {}'.format(args.low_resolution, args.image_size, epoch))
    wandb.log({'training y {} -> {}'.format(args.low_resolution, args.image_size):fig})
    save_image(y[:5].cpu(), args.output_dir + 'training_y_{}-{}{}.png'.format(args.low_resolution, args.image_size, epoch))
    
    fig = make_grid(x[:5], num_images=5, title='training image {}x{}, num epochs: {}'.format(args.image_size, args.image_size, epoch))
    wandb.log({'training image {}x{}'.format(args.image_size, args.image_size):fig})
    save_image(x[:5].cpu(), args.output_dir + 'training_image_{}x{}{}.png'.format(args.image_size, args.image_size, epoch))
    
    fig = make_grid(x_hat[:5], num_images=5, title='UNET train image reconstruction {}x{}, num epochs: {}'.format(args.image_size, args.image_size, epoch))
    wandb.log({'UNET train image reconstruction {}x{}'.format(args.image_size, args.image_size):fig})
    save_image(x_hat[:5].cpu(), args.output_dir + 'UNET_train_image_reconstruction_{}x{}{}.png'.format(args.image_size, args.image_size, epoch))
    
    # Val image logging and saving
    fig = make_grid(x_mask_val[:5], num_images=5, title='val mask, num epochs: {}'.format(epoch))
    wandb.log({'val mask':fig})   
    save_image(x_mask_val[:5].cpu(), args.output_dir + 'val_mask{}.png'.format(epoch))
    
    fig = make_grid(x_mask_hat_val[:5], num_images=5, title='UNET val mask reconstruction, num epochs: {}'.format(epoch))
    wandb.log({'UNET val mask reconstruction':fig})
    save_image(x_mask_hat_val[:5].cpu(), args.output_dir + 'UNET_val_mask_reconstruction{}.png'.format(epoch))
    
    fig = make_grid(y_val[:5], num_images=5, title='val y {} -> {}, num epochs: {}'.format(args.low_resolution, args.image_size, epoch))
    wandb.log({'val y {} -> {}'.format(args.low_resolution, args.image_size):fig})
    save_image(y_val[:5].cpu(), args.output_dir + 'val_y_{}-{}{}.png'.format(args.low_resolution, args.image_size, epoch))
    
    fig = make_grid(x_val[:5], num_images=5, title='val image {}x{}, num epochs: {}'.format(args.image_size, args.image_size, epoch))
    wandb.log({'val image {}x{}'.format(args.image_size, args.image_size):fig})
    save_image(x_val[:5].cpu(), args.output_dir + 'val_image_{}x{}{}.png'.format(args.image_size, args.image_size, epoch))
    
    fig = make_grid(x_hat_val[:5], num_images=5, title='UNET val image reconstruction {}x{}, num epochs: {}'.format(args.image_size, args.image_size, epoch))
    wandb.log({'UNET val image reconstruction {}x{}'.format(args.image_size, args.image_size):fig})
    save_image(x_hat_val[:5].cpu(), args.output_dir + 'UNET_val_image_reconstruction_{}x{}{}.png'.format(args.image_size, args.image_size, epoch))
    
    # Save
    print("Saving...")
    if (epoch+1) % args.num_epochs_save == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.output_dir + 'UNET_model_iter{}.pth'.format(epoch))
        

