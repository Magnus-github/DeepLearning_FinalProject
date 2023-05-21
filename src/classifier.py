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

IMAGE_SIZE = 128

class Classifier(nn.Module):
    """Baseline module for object classification."""

    def __init__(self, classification_mode="binary") -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Classifier, self).__init__()

        self.classification_mode = classification_mode

        self.features = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1).features  
        # output of mobilenet_v2 will be 1280x23x40 for 720x1280 input images
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images


        self.img_height = 224  # 720#480.0
        self.img_width = 224  # 1280#640.0

        # do dry run to determine output size of the backbone
        test_inp = torch.ones((1,3,self.img_height, self.img_width))
        test_out = self.features(test_inp) # 1x2560x7x7
        print("CLASSIFICATION MODE: ", self.classification_mode)
        if self.classification_mode == "binary":
            out_classes = 2
        elif self.classification_mode == "multi_class":
            out_classes = 37

        print(test_out.size())

        self.head = nn.Linear(nn.Flatten(-3, -1)(test_out).size()[1], out_classes)

        for child in self.features.children():
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
      w,h = image.size
      

      transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.RandomCrop((int(h-0.05*h),int(w-0.05*w))),
         transforms.Resize((224, 224), antialias=True),
         transforms.RandomHorizontalFlip(p=0.5),
        #  transforms.RandomRotation(20),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      transformed = transform(image)
      return transformed
    
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
#     Classifier()