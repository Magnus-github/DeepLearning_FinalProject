"""Baseline detector model.

Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
from typing import List, Optional, Tuple, TypedDict

import PIL
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights


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
        else:
            out_classes = 37

        self.head = nn.Linear(nn.Flatten(-3, -1)(test_out).size()[1], out_classes)

        # 1280x15x20 -> 5x15x20, where each element 5 channel tuple corresponds to
        #   (rel_x_offset, rel_y_offset, rel_x_width, rel_y_height, confidence
        # Where rel_x_offset, rel_y_offset is relative offset from cell_center
        # Where rel_x_width, rel_y_width is relative to image size
        # Where confidence is predicted IOU * probability of object center in this cell
        # self.out_cells_x = test_out.shape[1]  # 20 #40
        # self.out_cells_y = test_out.shape[2]  # 15 #23
        

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
        print(features_flat.size())
        out = self.head(features_flat)  # out size: n_batch x 2

        out = torch.nn.functional.softmax(out)

        return out



    def input_transform_training(self, image: Image) -> Tuple[torch.Tensor]:
        def crop_image(image):
            """Crop the images so only a specific region of interest is shown to my PyTorch model"""
            splitxL = 0.5
            splitxR = 0.51

            splityD = 0.5
            splityU = 0.51

            image = image[:, int(image.shape[1] * splityD):int(image.shape[1] * splityU),
                    int(image.shape[2] * splitxL):int(image.shape[2] * splitxR)]

            return image




        transform = transforms.Compose([
             transforms.PILToTensor(),
             transforms.ConvertImageDtype(torch.float),
             transforms.Lambda(crop_image),
             transforms.Resize((224, 224), antialias=True),
             #transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
        transformed = transform(image)


        return transformed






    def input_transform_testing(self, image: Image) -> Tuple[torch.Tensor]:
        def crop_image(image):
            """Crop the images so only a specific region of interest is shown to my PyTorch model"""
            splitxL = 0.5
            splitxR = 0.51

            splityD = 0.5
            splityU = 0.51

            image = image[:, int(image.shape[1] * splityD):int(image.shape[1] * splityU),
                    int(image.shape[2] * splitxL):int(image.shape[2] * splitxR)]

            return image




        transform = transforms.Compose([
             transforms.PILToTensor(),
             transforms.ConvertImageDtype(torch.float),
             #transforms.Lambda(crop_image),
             transforms.Resize((224, 224), antialias=True),
             #transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
        transformed = transform(image)


        return transformed
