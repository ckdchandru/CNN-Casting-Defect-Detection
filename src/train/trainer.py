from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.train.models import EpochMetrics, FitResult

DEFECT_LABEL = 1


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_path: Path,
        patience: int = 5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int) -> FitResult:
        result = FitResult()
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                val_pr_auc=val_metrics["pr_auc"],
            )
            result.history.append(epoch_metrics)

            improved = val_metrics["pr_auc"] > result.best_val_pr_auc
            if improved:
                result.best_epoch = epoch
                result.best_val_pr_auc = val_metrics["pr_auc"]
                result.best_val_recall = val_metrics["recall"]
                result.best_val_f1 = val_metrics["f1"]
                result.best_val_precision = val_metrics["precision"]
                self._save_checkpoint()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                result.stopped_early = True
                break

        result.checkpoint_path = str(self.checkpoint_path)
        return result

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    def _evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_defect_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_defect_probs.extend(probs[:, DEFECT_LABEL].cpu().tolist())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[DEFECT_LABEL], average=None, zero_division=0
        )
        pr_auc = average_precision_score(
            [1 if label == DEFECT_LABEL else 0 for label in all_labels],
            all_defect_probs,
        )

        return {
            "loss": total_loss / len(loader.dataset),
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "pr_auc": float(pr_auc),
        }

    def _save_checkpoint(self) -> None:
        """Save model state dict for the current best epoch."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)
