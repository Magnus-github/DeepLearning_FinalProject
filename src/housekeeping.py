import torch
from torchvision import transforms
from PIL import Image
import os


def housekeeping(root_dir="data/images/",output_dir= "data/images/"):
	transform = transforms.Compose([
		transforms.PILToTensor(),
         transforms.ConvertImageDtype(torch.float),
         transforms.Resize((224, 224), antialias=True),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


	files = [f'{root_dir}{file}' for file in os.listdir(root_dir)]


	for file in files:


		try:
			transformed_image = transform(Image.open(file))

		except:
			print("removed")
			os.remove(file)
			

