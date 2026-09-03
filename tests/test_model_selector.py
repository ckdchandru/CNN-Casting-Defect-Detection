import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.model_selector import ModelSelector
from src.train.models import FitResult


def _make_result(recall: float, pr_auc: float, f1: float) -> FitResult:
    result = FitResult()
    result.best_val_recall = recall
    result.best_val_pr_auc = pr_auc
    result.best_val_f1 = f1
    return result


def test_recall_decides_when_different():
    a = _make_result(recall=0.70, pr_auc=0.90, f1=0.90)
    b = _make_result(recall=0.80, pr_auc=0.60, f1=0.60)

    selection = ModelSelector().select(a, b)
    assert selection.winner == "B"
    assert "recall" in selection.reason


def test_pr_auc_breaks_recall_tie():
    a = _make_result(recall=0.75, pr_auc=0.85, f1=0.50)
    b = _make_result(recall=0.75, pr_auc=0.90, f1=0.40)

    selection = ModelSelector().select(a, b)
    assert selection.winner == "B"
    assert "pr_auc" in selection.reason


def test_f1_breaks_recall_and_pr_auc_tie():
    a = _make_result(recall=0.75, pr_auc=0.85, f1=0.60)
    b = _make_result(recall=0.75, pr_auc=0.85, f1=0.55)

    selection = ModelSelector().select(a, b)
    assert selection.winner == "A"
    assert "f1" in selection.reason


def test_exact_tie_defaults_to_a():
    a = _make_result(recall=0.75, pr_auc=0.85, f1=0.60)
    b = _make_result(recall=0.75, pr_auc=0.85, f1=0.60)

    selection = ModelSelector().select(a, b)
    assert selection.winner == "A"
    assert "tie" in selection.reason


def test_b_can_lose_even_with_weighting():
    # A can win too.
    a = _make_result(recall=0.90, pr_auc=0.90, f1=0.90)
    b = _make_result(recall=0.50, pr_auc=0.50, f1=0.50)

    selection = ModelSelector().select(a, b)
    assert selection.winner == "A"


if __name__ == "__main__":
    test_recall_decides_when_different()
    test_pr_auc_breaks_recall_tie()
    test_f1_breaks_recall_and_pr_auc_tie()
    test_exact_tie_defaults_to_a()
    test_b_can_lose_even_with_weighting()
    print("All model-selection tests passed.")
