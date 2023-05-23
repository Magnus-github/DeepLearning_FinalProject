import random
import torch
import torchvision.transforms as transforms
from PIL import Image


class CustomCrop(object):
        def __init__(self, cut) -> None:
                self.cut = cut
        
        def __call__(self, image: torch.Tensor) -> torch.Tensor:
            splitxL = self.cut * random.random()
            splitxR = (1-self.cut) + self.cut * random.random()

            splityD = self.cut * random.random()
            splityU = (1-self.cut) + self.cut * random.random()

            image = image[:, int(image.shape[1] * splityD):int(image.shape[1] * splityU),
                    int(image.shape[2] * splitxL):int(image.shape[2] * splitxR)]

            return image
        

# if __name__ == "__main__":
#        with Image.open("data/images/Abyssinian_27.jpg") as im:
#         # im.show()
#         tf = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
#         itf = transforms.ToPILImage()
#         im = tf(im)
#         itf(im).show()
#         # im = gaussianblur(im, 5)
#         gausTF = transforms.Lambda(CustomCrop(0.1))
#         # test_tf = transforms.ColorJitter(hue=[0])
#         im= gausTF(im)
#     #     im.transpose(1,2,0)
#         # im = Image.fromarray(im, 'RGB')
#         im = itf(im)
#         im.show()