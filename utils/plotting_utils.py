import torch
import matplotlib.pyplot as plt
import numpy as np
from misc import merge

def plot_pytorch_images(images, num_images=5, title=""):
    if images.is_cuda:
        images = images.detach().cpu()
    images = images.permute(0,2,3,1)
    img = np.expand_dims(images,0)
    img = merge(img[0],[1,num_images])
    fig = plt.figure(figsize=(8*max(1, num_images-2),8))
    plt.imshow(img)
    plt.gray()
    plt.title(title)
    plt.show()
    return fig