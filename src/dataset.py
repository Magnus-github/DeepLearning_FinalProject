import torch
from torch.utils.data import Dataset
import os
from  PIL import Image
from torchvision import transforms


class Pets(Dataset):
    def __init__(self, root_dir="./data/images/", transform=None, classification_mode="multi_class") -> None:
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transform

        if classification_mode == "binary" or classification_mode == "multi_class":
            self.classification_mode = classification_mode
        else:
            raise ValueError("The classification mode should be either 'binary' or 'multi_class'!")

        self.labels = self.__getlabels__()
        self.unlabelled = False
        
    def __getlabels__(self):
        files = [file for file in os.listdir(self.root_dir)]
        files = [file.split(".")[0] for file in files]
        files = [file.split("_")[:-1] for file in files]
        files = ["_".join(file) for file in files]
        files_cats = [file+"_cat" for file in files if file[0].isupper()]
        files_dogs = [file+"_dog" for file in files if file[0].islower()]
        files_cats.extend(files_dogs)
        files = files_cats
        if self.classification_mode == "binary":
            files = [file.split("_")[-1] for file in files]
        labels = list(set(files))
        return dict(zip(labels, torch.arange(len(labels))))
    
    def __getfiles__(self):
        return [f'{self.root_dir}{file}' for file in os.listdir(self.root_dir)]

    def __getitem__(self, idx):
        file_name = self.__getfiles__()[idx]
        img = self.transform(Image.open(file_name))
        if self.unlabelled:
            label = "unlabelled"
        else:
            label = ('_').join(file_name.split('/')[-1].split('.')[0].split('_')[:-1])
            if label[0].isupper():
                label += "_cat"
            else:
                label += "_dog"
            if self.classification_mode == "binary":
                label = label[-3:]
        return img, self.labels[label]
    
    def __len__(self):
        return len(self.__getfiles__())
    
if __name__ == "__main__":
    print(Pets().labels)
