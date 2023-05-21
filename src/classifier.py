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
import augmentations
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
      
        #hi mom

      transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      transformed = transform(image)
      return transformed
    

    def rand_augment(self, image:Image) -> torch.Tensor:
        """
        Implement the randaugment algorithm.
        """
        N = 8 # number of augmentations to be applied (total numberof augmentations is K=14)
        M = 8 # intensity of augmentations (intensitiy 0 <= M <= 10) -> linear scaling

        # RandomHorizontalFlip: IMPLEMENTED (not to be used in randaugment!!!!)
        # RandomRotation: MEDIUM-HARD
        # GussianBlur: EASY (copy from image analysis)
        # RandomPosterize: IDK
        # Identity: EASY
        # RandomAdjustSharpness: IDK (rather MEDIUM-HARD); sharpness val: 0->blurred, 1->original, n>1->sharpened by factor n
        # TranslateX: EASY-MEDIUM (with torch: transforms.RandomAffine(degrees=0, translate(X,0))) (analog for Y)
        # RandomAutoContrast: IDK, just use torch.transforms.RandomAutoContrast
        # RandomSolarize: EASY (all pixels above a thresh should be inverted)
        # ShearX/Y: same as translate
        # RandomEqualize: MEDIUM (change color histogram)
        # Color: no specification in paper -> use colorJitter from torch
        # Brightness: same as color

        transform_pool = [transforms.RandomHorizontalFlip(p=M/10), transforms.RandomRotation((M*90)/10), transforms.GaussianBlur((5,5), sigma=(M*5)/10),
                          transforms.RandomSolarize((M*0.75)/10), transforms.Compose([transforms.ConvertImageDtype(torch.uint8),transforms.RandomPosterize((8*M)/10), transforms.ConvertImageDtype(torch.float)])]

        # transform_pool.append(transforms.Lambda())
        # ...

        transforms_to_use = random.choices(transform_pool, k=N)
        print(transforms_to_use)


        transform_list = [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
        
        for tf in transforms_to_use:
            transform_list.append(tf)

        transform_list.append(transforms.Resize((224,224), antialias=True))
        # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        

        transform = transforms.Compose(transform_list)

        return transform(image)
    

if __name__ == "__main__":
    print(os.listdir(os.curdir))
    with Image.open("./data/images/american_bulldog_87.jpg") as im:
        clf = Classifier()
        #tf_im = clf.rand_augment(im)
        # im.show()
        # print(tf(im))
        # print(tf_im)
        #tf = transforms.Compose([transforms.PILToTensor(), transforms.RandomEqualize(p=1)])
        #print(tf(im))
        im=np.array(im)
        print(im.shape)
        im = augmentations.gaussianblur(im, 2)
        im.show()
        #plt.imshow(tf_im.permute(1, 2, 0))
        plt.show()