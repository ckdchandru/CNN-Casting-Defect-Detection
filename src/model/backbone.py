import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")


def build_backbone(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    if name == "resnet18":
        return _build_resnet18(pretrained, num_classes)
    if name == "efficientnet_b0":
        return _build_efficientnet_b0(pretrained, num_classes)
    raise ValueError(f"Unsupported backbone: {name!r}")


def _build_resnet18(pretrained: bool, num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_efficientnet_b0(pretrained: bool, num_classes: int) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
