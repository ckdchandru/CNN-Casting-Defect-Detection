
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.evaluate.models import EvaluationMetrics

DEFECT_LABEL = 1


class TestEvaluator:

    def evaluate(
        self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
    ) -> EvaluationMetrics:
        """Full metric set for one set of predictions."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[DEFECT_LABEL], average=None, zero_division=0
        )
        pr_auc = average_precision_score(labels, probs)
        accuracy = accuracy_score(labels, preds)
        matrix = confusion_matrix(labels, preds, labels=[0, DEFECT_LABEL])

        return EvaluationMetrics(
            precision=float(precision[0]),
            recall=float(recall[0]),
            f1=float(f1[0]),
            pr_auc=float(pr_auc),
            accuracy=float(accuracy),
            confusion_matrix=matrix.tolist(),
        )
