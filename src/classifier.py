"""Baseline detector model.

Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
import augmentations as A
import os

import random
import matplotlib.pyplot as plt



class Classifier(nn.Module):
    """Baseline module for object classification."""

    def __init__(self, classification_mode="binary") -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Classifier, self).__init__()

        self.classification_mode = classification_mode

        self.features = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features       
        # output of mobilenet_v2 will be 1280x23x40 for 720x1280 input images
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images


        self.img_height = 224  # 720#480.0
        self.img_width = 224  # 1280#640.0

        # do dry run to determine output size of the backbone
        test_inp = torch.ones((1,3,self.img_height, self.img_width))
        test_out = self.features(test_inp)
        print("CLASSIFICATION MODE: ", self.classification_mode)
        if self.classification_mode == "binary":
            out_classes = 2
        elif self.classification_mode == "multi_class":
            out_classes = 37

        self.head = nn.Linear(nn.Flatten(-3, -1)(test_out).size()[1], out_classes)

        count=0
        for child in self.features.children():
            count+=1
            if count < 19:
                for param in list(child.parameters())[:]:
                    param.requires_grad = False
        

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Compute output of neural network from input.

        Args:
            inp: The input images. Shape (N, 3, H, W).

        Returns:
            The output tensor containing the class for the image (one hot encoded).
        """
        features = self.features(inp)
        features_flat = nn.Flatten(-3,-1)(features)
        out = self.head(features_flat)  # out size: n_batch x 2

        # out = torch.nn.functional.softmax(out)

        return out
    

    def input_transform(self, image: Image) -> Tuple[torch.Tensor]:
      """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image:
                The image loaded from the dataset.

        Returns:
            transform:
                The composition of transforms to be applied to the image.
        """

      transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomRotation(20),
        #  transforms.ColorJitter(brightness=0.5, contrast=0.2, hue=0.3),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      transformed = transform(image)
      return transformed
    

    def rand_augment(self, image:Image) -> torch.Tensor:
        """
        Implement the randaugment algorithm.
        """
        N = 5 # number of augmentations to be applied (total numberof augmentations is K=14)
        M = 8 # intensity of augmentations (intensitiy 0 <= M <= 10) -> linear scaling

        transform_pool = [
                          transforms.Lambda(A.CustomGaussianBlurr((5*M)/10)),
                          transforms.Lambda(A.CustomIdentity()),
                          transforms.Lambda(A.CustomRandomSolarize(threshold=((10-M)/10))),
                        #   transforms.Lambda(A.CustomEqualization(p=M/10)),
                          transforms.RandomRotation((M*90)/10),
                          transforms.Compose([transforms.ConvertImageDtype(torch.uint8),transforms.RandomPosterize((8*M)/10), transforms.ConvertImageDtype(torch.float)]),
                          transforms.RandomAdjustSharpness(M),
                          transforms.RandomAffine(degrees=0, translate=((0.1*M)/10, 0)), # translate X
                          transforms.RandomAffine(degrees=0, translate=(0, (0.1*M)/10)), # translate Y
                          transforms.RandomAffine(degrees=0, shear=((20*M)/10)), # shear X
                          transforms.RandomAffine(degrees=0, shear=(0, 0, -(20*M)/10, (20*M)/10)), # shear Y
                          transforms.RandomAutocontrast(p=M/10),
                          transforms.ColorJitter(brightness=(0.5*M)/10), # brightness
                          transforms.ColorJitter(hue=(0.5*M)/10)
                          ]
        

        transforms_to_use = random.choices(transform_pool, k=N)
        print(transforms_to_use)


        transform_list = [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
        
        transform_list.extend(transforms_to_use)

        transform_list.append(transforms.Resize((224,224), antialias=True))
        transform_list.append(transforms.Lambda(A.Cutout(n_holes=1, length=16)))
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        

        transform = transforms.Compose(transform_list)

        return transform(image)
    

    def test_transform(self, image: Image) -> Tuple[torch.Tensor]:
      """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image:
                The image loaded from the dataset.

        Returns:
            transform:
                The composition of transforms to be applied to the image.
        """
    

      transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      transformed = transform(image)
      return transformed
    

    def weak_FM_transform(self, img: Image) -> torch.Tensor:
        """
        Perform a weak augmentation on a batch of images. Note, that the images are
        already transforemd to tensors and the correct size!
        """
        transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomAffine(degrees=0, translate=(0.125,0.125)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])    
        return transform(img)
    

    def strong_FM_transform(self, img: Image) -> torch.Tensor:
        """
        Perform a strong augmentation (RandAugment) on a batch of images. Note, that the images are already transforemd to tensors and the correct size!
        """
        transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.RandAugment(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)

# if __name__ == "__main__":
#     print(os.listdir(os.curdir))
#     with Image.open("./data/images/beagle_19.jpg") as im:
#         clf = Classifier()
#         im.show()
#         im = clf.rand_augment(im)
#         itf = transforms.ToPILImage()
#         im = itf(im)
#         im.show()
