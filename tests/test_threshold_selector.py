import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluate.threshold_selector import ThresholdSelector
from src.evaluate.test_evaluator import TestEvaluator
from src.evaluate.baseline import MajorityBaseline

CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _synthetic_val():
    labels = np.array([0] * 100 + [1] * 20)
    probs = np.concatenate([
        np.random.RandomState(0).beta(2, 8, 100),
        np.random.RandomState(0).beta(8, 2, 20),
    ])
    return labels, probs


def test_chosen_threshold_meets_precision_floor():
    labels, probs = _synthetic_val()
    result = ThresholdSelector(precision_floor=0.5, candidates=CANDIDATES).select(labels, probs)
    assert result.chosen_precision >= 0.5


def test_chosen_threshold_maximizes_recall_among_eligible():
    labels, probs = _synthetic_val()
    result = ThresholdSelector(precision_floor=0.5, candidates=CANDIDATES).select(labels, probs)

    eligible_recalls = [row.recall for row in result.table if row.precision >= 0.5]
    assert result.chosen_recall == max(eligible_recalls)


def test_impossible_floor_raises():
    labels, probs = _synthetic_val()
    selector = ThresholdSelector(precision_floor=1.5, candidates=CANDIDATES)
    try:
        selector.select(labels, probs)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "impossible precision_floor must raise, not silently pick something"


def test_table_has_one_row_per_candidate():
    labels, probs = _synthetic_val()
    result = ThresholdSelector(precision_floor=0.1, candidates=CANDIDATES).select(labels, probs)
    assert len(result.table) == len(CANDIDATES)


def test_majority_baseline_never_predicts_defect():
    labels = np.array([0] * 80 + [1] * 20)
    metrics = MajorityBaseline().evaluate(labels)
    assert metrics.recall == 0.0


def test_perfect_predictions_give_perfect_metrics():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = TestEvaluator().evaluate(labels, preds, probs)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


if __name__ == "__main__":
    test_chosen_threshold_meets_precision_floor()
    test_chosen_threshold_maximizes_recall_among_eligible()
    test_impossible_floor_raises()
    test_table_has_one_row_per_candidate()
    test_majority_baseline_never_predicts_defect()
    test_perfect_predictions_give_perfect_metrics()
    print("All threshold/evaluation tests passed.")
