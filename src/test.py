from typing import List, Optional, Tuple, TypedDict

import PIL
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights




def crop_image(image: PIL.Image.Image) -> PIL.Image.Image:
	"""Crop the images so only a specific region of interest is shown to my PyTorch model"""
	left, top, width, height = 20, 80, 40, 60

	return transforms.functional.crop(image, left=left, top=top, width=width, height=height)

transform = transforms.Compose([
	transforms.Lambda(crop_image),
	transforms.PILToTensor(),
	transforms.ConvertImageDtype(torch.float),
	transforms.Resize((224, 224), antialias=True),
	# transforms.RandomHorizontalFlip(p=0.5),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformed = transform(image)

