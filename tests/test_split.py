import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.near_duplicate_detector import NearDuplicateDetector
from src.data.split import StratifiedGroupSplitter, assert_no_group_leakage


def _load_grouped_pool():
    project_root = Path(__file__).resolve().parent.parent
    pool_df = pd.read_csv(project_root / "data/splits/constructed_pool.csv")
    detector = NearDuplicateDetector()
    return detector.detect(pool_df)


def test_no_group_spans_multiple_splits():
    grouped_df = _load_grouped_pool()
    splitter = StratifiedGroupSplitter(
        train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42
    )
    merged_df, _ = splitter.split(grouped_df)
    assert_no_group_leakage(merged_df)


def test_split_is_reproducible():
    grouped_df = _load_grouped_pool()
    splitter_a = StratifiedGroupSplitter(0.70, 0.15, 0.15, seed=42)
    splitter_b = StratifiedGroupSplitter(0.70, 0.15, 0.15, seed=42)

    merged_a, _ = splitter_a.split(grouped_df)
    merged_b, _ = splitter_b.split(grouped_df)

    assert merged_a["split"].tolist() == merged_b["split"].tolist()


def test_split_sizes_are_non_trivial():
    grouped_df = _load_grouped_pool()
    splitter = StratifiedGroupSplitter(0.70, 0.15, 0.15, seed=42)
    merged_df, _ = splitter.split(grouped_df)

    counts = merged_df["split"].value_counts()
    for split_name in ("train", "val", "test"):
        assert counts.get(split_name, 0) > 50, f"{split_name} split is too small"


if __name__ == "__main__":
    test_no_group_spans_multiple_splits()
    test_split_is_reproducible()
    test_split_sizes_are_non_trivial()
    print("All split tests passed.")
