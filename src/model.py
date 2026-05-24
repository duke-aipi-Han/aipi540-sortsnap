import torch
from torch import nn
from torchvision import models

# helper function to build the model

def build_model(num_classes: int, freeze_backbone: bool = True) -> torch.nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
