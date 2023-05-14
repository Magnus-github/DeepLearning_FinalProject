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


class Classifier(nn.Module):
    """Baseline module for object classification."""

    def __init__(self, batch_size) -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Classifier, self).__init__()

        self.features = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features       
        # output of mobilenet_v2 will be 1280x23x40 for 720x1280 input images
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        self.features_flat = nn.Flatten(-3,-1)(self.features)

        self.head = nn.Linear(batch_size, 2)

        self.img_height = 192  # 720#480.0
        self.img_width = 192  # 1280#640.0

        # do dry run to determine output size of the backbone
        # test_inp = torch.ones((3,self.img_height, self.img_width))
        # test_out = self.features(test_inp)

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
        features_flat = self.features_flat(self.features(inp))
        out = self.head(features_flat)  # out size: n_batch x 2

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
         transforms.resize(224,224),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      
      return transform