from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.explain.failure_analyzer import FAILURE_CATEGORIES


class Phase6ReportWriter:

    def __init__(self, figures_dir: Path, reports_dir: Path):
        self.figures_dir = figures_dir
        self.reports_dir = reports_dir

    def save_failure_grid(self, failure_df: pd.DataFrame, filename: str) -> Path:
        n = len(failure_df)
        cols = min(5, max(n, 1))
        rows = (n + cols - 1) // cols if n else 1

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = [axes] if n <= 1 else axes.flatten()

        for ax, (_, row) in zip(axes, failure_df.iterrows()):
            image = Image.open(row["gradcam_path"])
            ax.imshow(image)
            ax.set_title(
                f"true={row['true_label']} pred={row['predicted_label']}\n"
                f"p_defect={row['probability_defect']:.3f}",
                fontsize=8,
            )
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")

        plt.suptitle("All test-set misclassifications - Grad-CAM overlay")
        plt.tight_layout()
        path = self.figures_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def save_failure_records(self, failure_df: pd.DataFrame) -> Path:
        columns = [
            "image_id", "true_label", "predicted_label", "probability_defect",
            "gradcam_path", "observed_attention_region", "failure_category",
        ]
        path = self.reports_dir / "phase6_failure_records.csv"
        failure_df[columns].to_csv(path, index=False)
        return path

    def write_report(self, failure_df: pd.DataFrame, grid_path: Path, records_path: Path) -> Path:
        n_total = len(failure_df)
        n_fp = len(failure_df[(failure_df["true_label"] == 0) & (failure_df["predicted_label"] == 1)])
        n_fn = len(failure_df[(failure_df["true_label"] == 1) & (failure_df["predicted_label"] == 0)])

        lines = [
            "# Phase 6 Failure Analysis Report",
            "",
            f"- Total misclassifications: {n_total}",
            f"- False positives (predicted defect, actually normal): {n_fp}",
            f"- False negatives (predicted normal, actually defect): {n_fn}",
            "",
        ]

        if n_total < 12:
            lines += [
                "## Adaptation note",
                f"Only {n_total} misclassifications exist on the entire test set - too",
                "few to populate all four failure-mode buckets with 3-5 representative",
                "examples each, as the design doc's template assumes for a harder task.",
                "Every failure is shown individually below instead of sampled.",
                "",
            ]

        lines += [
            "## Failure-mode categories (design doc section 8)",
        ]
        for code, description in FAILURE_CATEGORIES.items():
            lines.append(f"- **{code}**: {description}")

        lines += [
            "",
            f"## All misclassifications (grid): {grid_path}",
            f"## Structured records (for categorization): {records_path}",
            "",
            "## Categorization",
            "Pending human visual review of the grid above. Each row in the",
            "records CSV needs `observed_attention_region` and `failure_category`",
            "filled in by hand once the images have been examined.",
        ]

        path = self.reports_dir / "phase6_failure_analysis_report.md"
        path.write_text("\n".join(lines))
        return path
