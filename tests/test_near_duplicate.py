import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.near_duplicate_detector import NearDuplicateDetector


def _load_pool():
    project_root = Path(__file__).resolve().parent.parent
    return pd.read_csv(project_root / "data/splits/constructed_pool.csv")


def test_every_image_gets_a_group():
    pool_df = _load_pool()
    grouped_df = NearDuplicateDetector().detect(pool_df)
    assert grouped_df["group_id"].notna().all()
    assert len(grouped_df) == len(pool_df)


def test_groups_are_never_cross_class():
    pool_df = _load_pool()
    grouped_df = NearDuplicateDetector().detect(pool_df)

    classes_per_group = grouped_df.groupby("group_id")["class_name"].nunique()
    assert (classes_per_group == 1).all(), "a group must not mix classes"


def test_grouping_is_deterministic():
    pool_df = _load_pool()
    detector = NearDuplicateDetector()

    grouped_a = detector.detect(pool_df)
    grouped_b = detector.detect(pool_df)

    sizes_a = grouped_a.groupby("group_id").size().sort_values().tolist()
    sizes_b = grouped_b.groupby("group_id").size().sort_values().tolist()
    assert sizes_a == sizes_b


def test_minority_class_still_has_enough_groups():
    # Gate needs 200+ minority groups.
    pool_df = _load_pool()
    grouped_df = NearDuplicateDetector().detect(pool_df)

    defect_groups = grouped_df[grouped_df["class_name"] == "defect"]["group_id"].nunique()
    assert defect_groups >= 200, f"only {defect_groups} independent defect groups"


if __name__ == "__main__":
    test_every_image_gets_a_group()
    test_groups_are_never_cross_class()
    test_grouping_is_deterministic()
    test_minority_class_still_has_enough_groups()
    print("All near-duplicate tests passed.")
