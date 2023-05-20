import torch
from torchvision import transforms
from PIL import Image
import os
import PIL

def housekeeping(root_dir="data/images/",output_dir= "data/imagesAugmented/"):
	#[mean, std] = findmeanstd(root_dir)
	[mean,std] = [[0.4811, 0.4499, 0.3964],[0.2330, 0.2300, 0.2329]]

	def crop_image(image):
		"""Crop the images so only a specific region of interest is shown to my PyTorch model"""
		splitxL = 0.33333
		splitxR = 0.666666

		splityD = 0.3333
		splityU = 0.6666

		image = image[:,int(image.shape[1]*splityD):int(image.shape[1]*splityU),int(image.shape[2]*splitxL):int(image.shape[2]*splitxR)]

		return image


	transform = transforms.Compose([
		transforms.PILToTensor(),

         transforms.ConvertImageDtype(torch.float),
		 transforms.Lambda(crop_image),
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
			

