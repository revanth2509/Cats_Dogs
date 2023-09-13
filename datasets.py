import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch

class CustomeTransform():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.t1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize(500),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    def loader(self,path):
        load = ImageFolder(path, transform=self.t1)
        train,valid = random_split(load,[7,3])
        train_data = DataLoader(train,batch_size=self.batch_size,shuffle=True)
        valid_data = DataLoader(valid,batch_size=self.batch_size,shuffle=True)
        return train_data, valid_data
       
