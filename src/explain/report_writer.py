from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class Phase5ReportWriter:

    def __init__(self, figures_dir: Path, reports_dir: Path):
        self.figures_dir = figures_dir
        self.reports_dir = reports_dir

    def save_manifest(self, manifest_df: pd.DataFrame, gradcam_dir: Path) -> Path:
        path = gradcam_dir / "gradcam_manifest.csv"
        manifest_df.to_csv(path, index=False)
        return path

    def save_true_positive_grid(
        self, manifest_df: pd.DataFrame, n_samples: int, filename: str
    ) -> Path:
        true_positives = manifest_df[
            (manifest_df["true_label"] == 1) & (manifest_df["predicted_label"] == 1)
        ]
        n_samples = min(n_samples, len(true_positives))
        samples = true_positives.sample(n=n_samples, random_state=42)

        cols = min(5, n_samples)
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for ax, (_, row) in zip(axes, samples.iterrows()):
            image = Image.open(row["gradcam_path"])
            ax.imshow(image)
            ax.axis("off")
        for ax in axes[n_samples:]:
            ax.axis("off")

        plt.suptitle("True positives - Grad-CAM overlay (gate check)")
        plt.tight_layout()
        path = self.figures_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def write_report(
        self, manifest_df: pd.DataFrame, grid_path: Path, gate_verdict: str, gate_notes: str
    ) -> Path:
        """Counts + the stated qualitative-check limitation."""
        n_total = len(manifest_df)
        n_correct = manifest_df["correct"].sum()
        n_true_positive = len(
            manifest_df[(manifest_df["true_label"] == 1) & (manifest_df["predicted_label"] == 1)]
        )

        lines = [
            "# Phase 5 Grad-CAM Report",
            "",
            f"- Heatmaps generated: {n_total} (every test prediction)",
            f"- Correct predictions: {n_correct}",
            f"- True positives available for gate review: {n_true_positive}",
            f"- Sample grid for gate review: {grid_path}",
            "",
            "## What Grad-CAM does and does not prove",
            "Grad-CAM shows the spatial regions that contributed most to the",
            "model's activation for the predicted class. It does NOT prove",
            "causality, and does NOT prove the model is \"looking at the",
            "defect\" in any certified sense.",
            "",
            "No pixel-level ground-truth defect masks exist for this dataset,",
            "so the gate check (Phase 5.3) is a QUALITATIVE spatial-alignment",
            "sanity check on a sample of true positives, not a quantitative",
            "localization-accuracy metric.",
            "",
            "## Gate verdict (human review)",
            f"**{gate_verdict}.** {gate_notes}",
        ]
        path = self.reports_dir / "phase5_gradcam_report.md"
        path.write_text("\n".join(lines))
        return path
