from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def return_model():
    resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)

    #Freeze layers
    for param in resnet_model.parameters():
        param.requires_grad = False
        
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
    
    return resnet_model