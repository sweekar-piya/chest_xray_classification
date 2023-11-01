from torch import nn

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*28*28, 256),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.LeakyReLU()
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(self.conv_block_1(x))
        x = self.maxpool(self.conv_block_2(x))
        x = self.maxpool(self.conv_block_3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(self.fc2(x))
        return x
    
def return_model():
    return BasicCNN()