import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.data.dataset import CLASS_TO_LABEL
from src.model.losses import build_loss, compute_class_weights


def _write_fake_train_csv(path: Path, n_normal: int, n_defect: int) -> None:
    rows = [{"filepath": f"n{i}.jpg", "class_name": "normal"} for i in range(n_normal)]
    rows += [{"filepath": f"d{i}.jpg", "class_name": "defect"} for i in range(n_defect)]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_class_weights_match_locked_formula(tmp_path=Path("/tmp")):
    csv_path = tmp_path / "fake_train.csv"
    _write_fake_train_csv(csv_path, n_normal=80, n_defect=20)

    weights = compute_class_weights(csv_path, CLASS_TO_LABEL)

    n_total, n_classes = 100, 2
    expected_normal = n_total / (n_classes * 80)
    expected_defect = n_total / (n_classes * 20)

    assert abs(weights[CLASS_TO_LABEL["normal"]].item() - expected_normal) < 1e-6
    assert abs(weights[CLASS_TO_LABEL["defect"]].item() - expected_defect) < 1e-6

    csv_path.unlink()


def test_experiment_a_loss_is_unweighted():
    loss = build_loss("A")
    assert loss.weight is None


def test_experiment_b_loss_uses_given_weights():
    weights = torch.tensor([0.6, 2.4])
    loss = build_loss("B", weights)
    assert torch.equal(loss.weight, weights)


def test_experiment_b_without_weights_raises():
    try:
        build_loss("B", None)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Experiment B must require class_weights"


if __name__ == "__main__":
    test_class_weights_match_locked_formula()
    test_experiment_a_loss_is_unweighted()
    test_experiment_b_loss_uses_given_weights()
    test_experiment_b_without_weights_raises()
    print("All loss/weight tests passed.")
