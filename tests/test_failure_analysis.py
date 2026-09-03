import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.explain.failure_analyzer import FailureAnalyzer


def _fake_predictions():
    return pd.DataFrame({
        "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "true_label": [0, 1, 0, 1],
        "predicted_label": [0, 1, 1, 0],
        "probability_defect": [0.1, 0.9, 0.6, 0.3],
        "threshold": 0.5,
        "correct": [True, True, False, False],
    })


def _fake_manifest():
    return pd.DataFrame({
        "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "true_label": [0, 1, 0, 1],
        "predicted_label": [0, 1, 1, 0],
        "correct": [True, True, False, False],
        "gradcam_path": ["/g/a.png", "/g/b.png", "/g/c.png", "/g/d.png"],
    })


def test_analyze_extracts_only_misclassified():
    failure_df = FailureAnalyzer().analyze(_fake_predictions(), _fake_manifest())
    assert set(failure_df["image_id"]) == {"c.jpg", "d.jpg"}
    assert len(failure_df) == 2


def test_analyze_joins_gradcam_path():
    failure_df = FailureAnalyzer().analyze(_fake_predictions(), _fake_manifest())
    row_c = failure_df[failure_df["image_id"] == "c.jpg"].iloc[0]
    assert row_c["gradcam_path"] == "/g/c.png"


def test_upsert_preserves_existing_human_annotations(tmp_path=Path("/tmp")):
    records_path = tmp_path / "fake_phase6_records.csv"

    # Simulate a prior run where a human already categorized "c.jpg".
    existing = pd.DataFrame([{
        "image_id": "c.jpg",
        "true_label": 0,
        "predicted_label": 1,
        "probability_defect": 0.6,
        "gradcam_path": "/g/c.png",
        "observed_attention_region": "background rig marks",
        "failure_category": "A",
    }])
    existing.to_csv(records_path, index=False)

    analyzer = FailureAnalyzer()
    fresh_df = analyzer.analyze(_fake_predictions(), _fake_manifest())
    merged = analyzer.upsert_with_existing(fresh_df, records_path)

    row_c = merged[merged["image_id"] == "c.jpg"].iloc[0]
    assert row_c["failure_category"] == "A", "rerun must not erase a human annotation"
    assert row_c["observed_attention_region"] == "background rig marks"

    row_d = merged[merged["image_id"] == "d.jpg"].iloc[0]
    assert row_d["failure_category"] == "", "new misclassification should start blank"

    records_path.unlink()


def test_upsert_with_no_existing_file_returns_fresh(tmp_path=Path("/tmp")):
    records_path = tmp_path / "does_not_exist_yet.csv"
    if records_path.exists():
        records_path.unlink()

    analyzer = FailureAnalyzer()
    fresh_df = analyzer.analyze(_fake_predictions(), _fake_manifest())
    merged = analyzer.upsert_with_existing(fresh_df, records_path)

    assert len(merged) == 2
    assert (merged["failure_category"] == "").all()


if __name__ == "__main__":
    test_analyze_extracts_only_misclassified()
    test_analyze_joins_gradcam_path()
    test_upsert_preserves_existing_human_annotations()
    test_upsert_with_no_existing_file_returns_fresh()
    print("All failure-analysis tests passed.")
