from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class StratifiedGroupSplitter:

    def __init__(self, train_frac: float, val_frac: float, test_frac: float, seed: int):
        total = train_frac + val_frac + test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {total}")
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def split(self, grouped_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        group_table = (
            grouped_df.groupby("group_id")
            .agg(class_name=("class_name", "first"), n_images=("group_id", "size"))
            .reset_index()
        )

        group_table["split"] = self._split_groups(group_table)

        merged = grouped_df.merge(
            group_table[["group_id", "split"]], on="group_id", how="left"
        )

        diagnostics = self._diagnostics(merged)
        return merged, diagnostics

    def _split_groups(self, group_table: pd.DataFrame) -> pd.Series:
        # Force plain numpy arrays.
        # sklearn needs real indexing here.
        group_ids = group_table["group_id"].to_numpy(dtype=object)
        classes = group_table["class_name"].to_numpy(dtype=object)

        train_ids, temp_ids, train_classes, temp_classes = train_test_split(
            group_ids,
            classes,
            train_size=self.train_frac,
            stratify=classes,
            random_state=self.seed,
        )

        val_share_of_temp = self.val_frac / (self.val_frac + self.test_frac)
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_share_of_temp,
            stratify=temp_classes,
            random_state=self.seed,
        )

        split_map = {}
        split_map.update({g: "train" for g in train_ids})
        split_map.update({g: "val" for g in val_ids})
        split_map.update({g: "test" for g in test_ids})
        return group_table["group_id"].map(split_map)

    def _diagnostics(self, merged: pd.DataFrame) -> Dict[str, int]:
        """Counts per split, for the caller to print."""
        counts = merged.groupby(["split", "class_name"]).size()
        return counts.to_dict()


def assert_no_group_leakage(merged: pd.DataFrame) -> None:
    splits_per_group = merged.groupby("group_id")["split"].nunique()
    leaking = splits_per_group[splits_per_group > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} group(s) span multiple splits")
