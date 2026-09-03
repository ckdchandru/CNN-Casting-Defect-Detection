import os
from pathlib import Path

import pandas as pd


class Phase2SplitWriter:

    def __init__(self, splits_dir: Path):
        self.splits_dir = splits_dir

    def write(self, merged_df: pd.DataFrame) -> dict:
        paths = {}
        for split_name in ("train", "val", "test"):
            subset = merged_df[merged_df["split"] == split_name]
            out = subset[["filepath", "class_name"]].copy()
            out["filepath"] = out["filepath"].apply(
                lambda p: os.path.relpath(p, start=".")
            )
            path = self.splits_dir / f"{split_name}.csv"
            out.to_csv(path, index=False)
            paths[split_name] = path
        return paths

    def write_near_duplicate_report(
        self, grouped_df: pd.DataFrame, reports_dir: Path
    ) -> Path:
        group_sizes = grouped_df.groupby("group_id").size()
        multi_groups = group_sizes[group_sizes > 1]

        lines = [
            "# Phase 2 Near-Duplicate Report",
            "",
            "Perceptual-hash grouping (8x8 average hash, exact match,",
            "same class only) - catches augmented siblings that exact-MD5",
            "matching (Phase 1) cannot, since rotation/flip/jitter changes",
            "every byte but leaves the physical part visually identical.",
            "",
            f"- Total images: {len(grouped_df):,}",
            f"- Total groups: {grouped_df['group_id'].nunique():,}",
            f"- Groups with >1 image: {len(multi_groups):,}",
            f"- Images inside a multi-image group: {multi_groups.sum():,}",
            f"- Largest group: {group_sizes.max()} images",
            "",
            "## Per-class effective independent count",
        ]
        per_class = grouped_df.groupby("class_name")["group_id"].nunique()
        raw_counts = grouped_df.groupby("class_name").size()
        for class_name in per_class.index:
            lines.append(
                f"- {class_name}: {per_class[class_name]:,} groups "
                f"(from {raw_counts[class_name]:,} raw images)"
            )

        path = reports_dir / "phase2_near_duplicate_report.md"
        path.write_text("\n".join(lines))
        return path
