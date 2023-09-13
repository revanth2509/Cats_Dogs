import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_shape,
                6,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                6,
                16,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                16,
                24,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2,2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                input_shape*12,
                input_shape*24,
                kernel_size=2,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_shape*3*2*2*2),
            nn.MaxPool2d(2))


        self.fc1 = nn.Sequential(
            nn.Linear(24*60*60, 300),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(300,150),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(150,2),
            nn.LogSoftmax(dim=1))

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1,24*60*60)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


