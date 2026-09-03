from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_TO_LABEL, CastingDataset
from src.data.transforms import build_transforms
from src.model.backbone import build_backbone
from src.model.losses import build_loss, compute_class_weights
from src.train.models import FitResult
from src.train.trainer import Trainer


class ExperimentRunner:

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

    def run(self, experiment: str) -> FitResult:
        """Train Experiment A, B, or C to completion."""
        train_csv = self.config["data"]["splits_dir"] / "train.csv"
        val_csv = self.config["data"]["splits_dir"] / "val.csv"

        transform_pipelines = build_transforms(
            image_size=self.config["train"]["image_size"],
            augmentation_config=self.config["augmentation"],
        )
        train_dataset = CastingDataset(train_csv, transform=transform_pipelines["train"])
        val_dataset = CastingDataset(val_csv, transform=transform_pipelines["eval"])

        batch_size = self.config["train"]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = build_backbone(
            name=self.config["model"]["backbone"],
            pretrained=self.config["model"]["pretrained"],
            num_classes=self.config["model"]["num_classes"],
        ).to(self.device)

        class_weights = None
        if experiment in ("B", "C"):
            class_weights = compute_class_weights(train_csv, CLASS_TO_LABEL).to(self.device)

        criterion = build_loss(experiment, class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["train"]["learning_rate"])

        checkpoint_path = self.config["paths"]["checkpoints_dir"] / f"experiment_{experiment}.pt"
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            checkpoint_path=checkpoint_path,
            patience=self.config["train"]["early_stopping_patience"],
        )

        return trainer.fit(
            train_loader, val_loader, max_epochs=self.config["train"]["epochs"]
        )
