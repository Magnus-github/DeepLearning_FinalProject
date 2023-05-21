from typing import Any, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

import random
import matplotlib.pyplot as plt



class CustomGaussianBlurr(object):
    def __init__(self, sigma) -> None:
        self.sigma = sigma

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a Gaussian blurr with variance sigma to each color layer of the image.
        The input is expected to be a torch.Tensor with datatype torch.float.
        """
        im = image.numpy()
        [c, h, w] = np.shape(im)  #colour, height, width
        
        [x,y]=(np.meshgrid(np.linspace(0, 1-1/w, w),np.linspace(0, 1-1/h, h)))
        ffft = np.exp(self.sigma * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y) - 2))
        pixels=np.zeros((c,h,w))
        for i in range(c):
            pfft = np.fft.fft2(im[i,:,:])
            pixels[i,:,:]= np.real(np.fft.ifft2(ffft * pfft))

        pixels = torch.from_numpy(pixels)

        return pixels
    
class CustomRandomSolarize(object):
    def __init__(self, threshold, p=0.5) -> None:
        self.threshold = threshold
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Randomly solarize an image, i.e. invert those pixels that are above *threshold* with a probability *p*.
        The input image is expected to be a torch.float torch.Tensor which implies that the threshold needs to be between [0,1].
        threshold = 0 -> max intensity
        threshold = 1 -> min intensity
        """

        if random.random() < self.p:
            image = torch.where(image > self.threshold, 1 - image, image)     
        return image


class CustomIdentity(object):
    def __init__(self) -> None:
        pass

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image




        

if __name__ == "__main__":
    with Image.open("data/images/Abyssinian_27.jpg") as im:
        im.show()
        tf = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        itf = transforms.ToPILImage()
        im = tf(im)
        print(im.size())
        
        # im = gaussianblur(im, 5)
        gausTF = transforms.Lambda(CustomGaussianBlurr(5))
        test_tf = transforms.ColorJitter(hue=[0 ])
        im= test_tf(im)
    #     im.transpose(1,2,0)
        # im = Image.fromarray(im, 'RGB')
        im = itf(im)
        im.show()
        
    #     #plt.imshow(tf_im.permute(1, 2, 0))
    #     #plt.show()