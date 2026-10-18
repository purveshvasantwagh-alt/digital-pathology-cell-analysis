from torchvision import models
from torch import nn


def build_model(num_classes: int, freeze_backbone: bool = True):
    """Create an ImageNet-pretrained ResNet18 for the requested classes."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model