import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.helpers import full_path

CATEGORIES = {
    "A": "Background/acquisition artifact",
    "B": "Correct region, weak defect",
    "C": "Visual ambiguity (class confusion)",
    "D": "Localization failure",
}

RECORD_COLUMNS = [
    "image_id", "true_label", "predicted_label", "probability_defect",
    "gradcam_path", "observed_attention_region", "failure_category",
]


def find_failures(predictions_df, manifest_df):
    wrong = predictions_df[~predictions_df["correct"]].copy()
    wrong = wrong.merge(manifest_df[["image_id", "gradcam_path"]], on="image_id", how="left")

    # manual review later
    wrong["observed_attention_region"] = ""
    wrong["failure_category"] = ""
    return wrong


def keep_old_annotations(new_df, records_path):
    if not records_path.exists():
        return new_df

    old_df = pd.read_csv(records_path)
    done = old_df[old_df["failure_category"].fillna("") != ""]
    not_done = new_df[~new_df["image_id"].isin(done["image_id"])]
    return pd.concat([done, not_done], ignore_index=True)


def count_fp_fn(failure_df):
    fp = ((failure_df["true_label"] == 0) & (failure_df["predicted_label"] == 1)).sum()
    fn = ((failure_df["true_label"] == 1) & (failure_df["predicted_label"] == 0)).sum()
    return int(fp), int(fn)


def failure_grid(failure_df, path):
    n = len(failure_df)
    cols = min(5, max(n, 1))
    rows = max((n + cols - 1) // cols, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, failure_df.iterrows()):
        ax.imshow(Image.open(full_path(row["gradcam_path"])))
        ax.set_title(
            f"true={row['true_label']} pred={row['predicted_label']}\n"
            f"p_defect={row['probability_defect']:.3f}",
            fontsize=8,
        )

    plt.suptitle("All test-set misclassifications - Grad-CAM overlay")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_phase6_report(failure_df, grid_path, records_path, path):
    fp, fn = count_fp_fn(failure_df)
    total = len(failure_df)

    lines = [
        "# Phase 6 Failure Analysis Report",
        "",
        f"- Total misclassifications: {total}",
        f"- False positives (predicted defect, actually normal): {fp}",
        f"- False negatives (predicted normal, actually defect): {fn}",
        "",
    ]
    if total < 12:
        lines += [
            "## Adaptation note",
            f"Only {total} misclassifications exist on the entire test set - too",
            "few to fill all four failure-mode buckets with 3-5 examples each.",
            "Every failure is shown individually below instead of sampled.",
            "",
        ]

    lines.append("## Failure-mode categories")
    for code, text in CATEGORIES.items():
        lines.append(f"- **{code}**: {text}")

    lines += [
        "",
        f"## All misclassifications (grid): {grid_path}",
        f"## Structured records (for categorization): {records_path}",
        "",
        "## Categorization",
        "Each row in the records CSV needs `observed_attention_region` and",
        "`failure_category` filled in by hand after looking at the grid.",
    ]
    path.write_text("\n".join(lines))
