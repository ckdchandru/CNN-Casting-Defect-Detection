import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from src.data.dataset import LABELS


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # train layer4 only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(config, experiment, device):
    model = build_model(config["model"]["num_classes"], pretrained=False).to(device)
    checkpoint = config["paths"]["checkpoints"] / f"experiment_{experiment}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model, checkpoint


def get_class_weights(train_csv):
    train_df = pd.read_csv(train_csv)
    counts = train_df["class_name"].value_counts()
    n_classes = len(LABELS)

    weights = torch.zeros(n_classes)
    for name, label in LABELS.items():
        n = counts.get(name, 0)
        if n == 0:
            raise ValueError(f"class {name} has zero train samples")
        # balanced weighting
        weights[label] = len(train_df) / (n_classes * n)
    return weights


def get_loss(experiment, class_weights=None):
    if experiment == "A":
        return nn.CrossEntropyLoss()
    if experiment == "B":
        if class_weights is None:
            raise ValueError("Experiment B needs class weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"Unknown experiment: {experiment}")
