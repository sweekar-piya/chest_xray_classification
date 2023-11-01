import torch
from torch import nn
from src.utils.constants import MODEL_DIR

class FirstModel(nn.Module):
    def __init__(self):
        super(FirstModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(128*28*28, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.conv_block_1(x))
        x = self.pool(self.conv_block_2(x))
        x = self.pool(self.conv_block_3(x))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x

def return_model():
    return FirstModel()