from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    train_csv_path: Path, class_to_label: Dict[str, int]
) -> torch.Tensor:
    train_df = pd.read_csv(train_csv_path)
    counts = train_df["class_name"].value_counts()

    n_train = len(train_df)
    n_classes = len(class_to_label)

    weights = torch.zeros(n_classes)
    for class_name, label in class_to_label.items():
        n_class = counts.get(class_name, 0)
        if n_class == 0:
            raise ValueError(f"class {class_name!r} has zero train samples")
        weights[label] = n_train / (n_classes * n_class)

    return weights


def build_loss(experiment: str, class_weights: torch.Tensor = None) -> nn.Module:
    if experiment == "A":
        return nn.CrossEntropyLoss()
    if experiment == "B":
        if class_weights is None:
            raise ValueError("Experiment B requires class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    if experiment == "C":
        return FocalLoss(gamma=2.0, weight=class_weights)
    raise ValueError(f"Unknown experiment: {experiment!r}")


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - target_probs) ** self.gamma
        loss = -focal_term * target_log_probs

        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights

        return loss.mean()
