from pathlib import Path

import pandas as pd

FAILURE_CATEGORIES = {
    "A": "Background/acquisition artifact",
    "B": "Correct region, weak defect",
    "C": "Visual ambiguity (class confusion)",
    "D": "Localization failure",
}


class FailureAnalyzer:

    def analyze(self, predictions_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
        """False positives and false negatives, with heatmap path."""
        misclassified = predictions_df[predictions_df["correct"] == False].copy()

        merged = misclassified.merge(
            manifest_df[["image_id", "gradcam_path"]], on="image_id", how="left"
        )

        # Human fills these in later.
        merged["observed_attention_region"] = ""
        merged["failure_category"] = ""

        return merged

    def upsert_with_existing(self, fresh_df: pd.DataFrame, records_path: Path) -> pd.DataFrame:
        if not records_path.exists():
            return fresh_df

        existing_df = pd.read_csv(records_path)
        already_annotated = existing_df[existing_df["failure_category"].fillna("") != ""]

        new_only = fresh_df[~fresh_df["image_id"].isin(already_annotated["image_id"])]
        return pd.concat([already_annotated, new_only], ignore_index=True)
