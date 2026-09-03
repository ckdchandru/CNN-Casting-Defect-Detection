import numpy as np

from src.evaluate.models import EvaluationMetrics
from src.evaluate.test_evaluator import TestEvaluator

NORMAL_LABEL = 0


class MajorityBaseline:

    def evaluate(self, labels: np.ndarray) -> EvaluationMetrics:
        """Always predicts normal, never flags a defect."""
        preds = np.full_like(labels, fill_value=NORMAL_LABEL)
        probs = np.zeros_like(labels, dtype=float)

        return TestEvaluator().evaluate(labels, preds, probs)
