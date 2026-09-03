from dataclasses import dataclass, field
from typing import List


@dataclass
class EpochMetrics:

    epoch: int
    train_loss: float
    val_loss: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_pr_auc: float


@dataclass
class FitResult:

    history: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_recall: float = 0.0
    best_val_pr_auc: float = 0.0
    best_val_f1: float = 0.0
    best_val_precision: float = 0.0
    checkpoint_path: str = ""
    stopped_early: bool = False
