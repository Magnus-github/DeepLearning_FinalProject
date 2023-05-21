from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

import random
import matplotlib.pyplot as plt


def gaussianblur(inpic, sigma):
    pfft = np.fft.fft2(inpic)
    [h, w, c] = np.shape(inpic)  #height, width, colout
    pixels=np.empty((h,w,c))
    for i in range(1):
        [x,y]=(np.meshgrid(np.linspace(0, 1-1/w, w),np.linspace(0, 1-1/h, h)))
        ffft = np.exp(sigma * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) - 2))
        pixels[:,:,i]= np.real(np.fft.ifft2(ffft * pfft[:,:,i]))
    return pixels
        

if __name__ == "__main__":
    with Image.open("data\images\Abyssinian_10.jpg") as im:

        im = gaussianblur(im, 2)
        im.transpose(0,1,2)
        im = Image.fromarray(im, 'RGB')
        im.show()
        
        #plt.imshow(tf_im.permute(1, 2, 0))
        #plt.show()
