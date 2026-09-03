import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.explain.failure_analyzer import FailureAnalyzer


def _fake_predictions():
    return pd.DataFrame({
        "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "true_label": [0, 0, 1, 1],
        "predicted_label": [0, 1, 1, 0],
        "probability_defect": [0.1, 0.6, 0.9, 0.3],
        "threshold": [0.5, 0.5, 0.5, 0.5],
        "correct": [True, False, True, False],
    })


def _fake_manifest():
    return pd.DataFrame({
        "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "true_label": [0, 0, 1, 1],
        "predicted_label": [0, 1, 1, 0],
        "correct": [True, False, True, False],
        "gradcam_path": ["a_cam.png", "b_cam.png", "c_cam.png", "d_cam.png"],
    })


def test_only_misclassifications_kept():
    failure_df = FailureAnalyzer().analyze(_fake_predictions(), _fake_manifest())
    assert len(failure_df) == 2
    assert set(failure_df["image_id"]) == {"b.jpg", "d.jpg"}


def test_gradcam_path_joined_correctly():
    failure_df = FailureAnalyzer().analyze(_fake_predictions(), _fake_manifest())
    row_b = failure_df[failure_df["image_id"] == "b.jpg"].iloc[0]
    assert row_b["gradcam_path"] == "b_cam.png"


def test_placeholder_columns_present_and_empty():
    failure_df = FailureAnalyzer().analyze(_fake_predictions(), _fake_manifest())
    assert "observed_attention_region" in failure_df.columns
    assert "failure_category" in failure_df.columns
    assert (failure_df["observed_attention_region"] == "").all()


def test_no_misclassifications_gives_empty_df():
    all_correct = _fake_predictions().copy()
    all_correct["correct"] = True
    failure_df = FailureAnalyzer().analyze(all_correct, _fake_manifest())
    assert len(failure_df) == 0


if __name__ == "__main__":
    test_only_misclassifications_kept()
    test_gradcam_path_joined_correctly()
    test_placeholder_columns_present_and_empty()
    test_no_misclassifications_gives_empty_df()
    print("All failure analysis tests passed.")
