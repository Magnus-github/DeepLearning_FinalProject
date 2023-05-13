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
    """Baseline module for object detection."""

    def __init__(self) -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Classifier, self).__init__()

        self.features = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features       
        # output of mobilenet_v2 will be 1280x23x40 for 720x1280 input images
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        self.features_flat = nn.Flatten()(self.features)
        self.head = nn.Linear(torch.Size(self.features_flat)[0], 2)

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
            The output tensor encoding the predicted bounding boxes.
            Shape (N, 5, self.out_cells_y, self.out_cells_y).
        """
        features_flat = self.features_flat(self.features(inp))
        out = self.head(features_flat)  # Linear (i.e., no) activation

        return out
    

    def input_transform(self, image: Image, anns: List) -> Tuple[torch.Tensor]:
      """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image:
                The image loaded from the dataset.
            anns:
                List of annotations in COCO format.

        Returns:
            Tuple:
                image: The image. Shape (3, H, W).
                target:
                    The network target encoding the bounding box.
                    Shape (5, self.out_cells_y, self.out_cells_x).
        """
      transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.resize(192,192),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

      transformed = transform(image)
      image = transformed["image"]

      target = torch.zeros(2,1)
      
      return image, target

    def input_transform_for_training(self, image: Image, anns: List) -> Tuple[torch.Tensor]:
      bbs = [[ann["bbox"][0], ann["bbox"][1], ann["bbox"]
              [2], ann["bbox"][3], "idk"] for ann in anns]
      transform = A.Compose([
          A.Resize(height=480, width=640),
          A.HorizontalFlip(p=0.5),
          A.MotionBlur(p=0.01),
          A.CLAHE(p=0.01),
          A.ColorJitter(p=0.01),
          A.Emboss(p=0.01),
          A.FancyPCA(p=0.01),
          A.GaussNoise(p=0.01),
          A.HueSaturationValue(p=0.01),
          A.ISONoise(p=0.01),
          A.PixelDropout(p=0.01),
          A.RandomBrightness(p=0.01),
          A.RandomContrast(p=0.01),
          A.RandomSunFlare(p=0.01),
          A.RandomShadow(p=0.1),
          A.Sharpen(p=0.1),
          A.augmentations.geometric.transforms.Affine(shear=[-45, 45], p=0.01),
      ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1))
      transformed = transform(image=np.asarray(image), bboxes=bbs)
      image = transformed["image"]
      bbs = transformed["bboxes"]

      # The image has to be normalized, this has to be done AFTER transforming the bounding boxes,
      # otherwise they get messed up
      image = transforms.ToTensor()(image)
      image = transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

      target = torch.zeros(5, self.out_cells_y, self.out_cells_x)
      for bbx in bbs:
        x = bbx[0]
        y = bbx[1]
        width = bbx[2]
        height = bbx[3]
        x_center = x + width / 2.0
        y_center = y + height / 2.0
        x_center_rel = x_center / self.img_width * self.out_cells_x
        y_center_rel = y_center / self.img_height * self.out_cells_y
        x_ind = int(x_center_rel)
        y_ind = int(y_center_rel)
        x_cell_pos = x_center_rel - x_ind
        y_cell_pos = y_center_rel - y_ind
        rel_width = width / self.img_width
        rel_height = height / self.img_height
        # channels, rows (y cells), cols (x cells)
        target[4, y_ind, x_ind] = 1
        # bb size
        target[0, y_ind, x_ind] = x_cell_pos
        target[1, y_ind, x_ind] = y_cell_pos
        target[2, y_ind, x_ind] = rel_width
        target[3, y_ind, x_ind] = rel_height

      return image, target
