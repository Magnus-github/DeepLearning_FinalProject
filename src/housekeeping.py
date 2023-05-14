import torch
from torchvision import transforms
from PIL import Image
import os

transform = transforms.Compose([
         transforms.PILToTensor(),
         transforms.Resize((224,224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

def housekeeping(root_dir="data/images/", transform = transform):
    files = [f'{root_dir}{file}' for file in os.listdir(root_dir)]
    for file in files:
        try:
            transform(Image.open(file))
        except:
            os.remove(file)
