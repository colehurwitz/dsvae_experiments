import torch
import matplotlib.pyplot as plt
import numpy as np
from misc import merge
import torchvision.utils as vutils

def plot_pytorch_images(images, num_images=5, title="", xticks=False, yticks=False):
    images = images.detach()
    if images.is_cuda:
        images = images.cpu()
    images = images.permute(0,2,3,1)
    img = np.expand_dims(images,0)
    img = merge(img[0],[1,num_images])
    fig = plt.figure(figsize=(8*max(1, num_images-2),8))
    plt.imshow(img)
    plt.gray()
    plt.title(title)
    if not xticks:
        plt.xticks([],[])
    if not yticks:
        plt.yticks([],[])
    plt.show()
    return fig

def make_grid(images, num_images=5, title="", axis="off", size=(12,12)):
    images = images.detach()
    if images.is_cuda:
        images = images.cpu()
    fig = plt.figure(figsize=size)
    plt.axis(axis)
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=False).cpu(),(1,2,0)))
    plt.show()
    return fig