from dataclasses import dataclass, field
from typing import List


@dataclass
class ThresholdRow:

    threshold: float
    precision: float
    recall: float


@dataclass
class ThresholdResult:

    chosen_threshold: float
    chosen_precision: float
    chosen_recall: float
    precision_floor: float
    table: List[ThresholdRow] = field(default_factory=list)


@dataclass
class EvaluationMetrics:

    precision: float
    recall: float
    f1: float
    pr_auc: float
    accuracy: float
    confusion_matrix: List[List[int]]
