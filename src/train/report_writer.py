from pathlib import Path

import matplotlib.pyplot as plt

from src.train.model_selector import SelectionResult
from src.train.models import FitResult


class Phase3ReportWriter:

    def __init__(self, figures_dir: Path, reports_dir: Path):
        self.figures_dir = figures_dir
        self.reports_dir = reports_dir

    def save_training_curve(self, result: FitResult, experiment: str) -> Path:
        epochs = [m.epoch for m in result.history]
        recall = [m.val_recall for m in result.history]
        pr_auc = [m.val_pr_auc for m in result.history]
        f1 = [m.val_f1 for m in result.history]

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, recall, label="val_recall", marker="o")
        plt.plot(epochs, pr_auc, label="val_pr_auc", marker="o")
        plt.plot(epochs, f1, label="val_f1", marker="o")
        plt.axvline(result.best_epoch, color="gray", linestyle="--", label="best epoch")
        plt.title(f"Experiment {experiment} - validation metrics")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()

        path = self.figures_dir / f"phase3_experiment_{experiment}_curve.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def write_comparison_report(
        self,
        result_a: FitResult,
        result_b: FitResult,
        selection: SelectionResult,
    ) -> Path:
        lines = [
            "# Phase 3 Model Selection Report",
            "",
            f"**Winner: Experiment {selection.winner}.** {selection.reason}",
            "",
            "## Experiment A (naive, unweighted CE)",
            f"- Best epoch: {result_a.best_epoch} (stopped early: {result_a.stopped_early})",
            f"- Recall: {result_a.best_val_recall:.4f}",
            f"- PR-AUC: {result_a.best_val_pr_auc:.4f}",
            f"- F1: {result_a.best_val_f1:.4f}",
            f"- Precision: {result_a.best_val_precision:.4f}",
            "",
            "## Experiment B (weighted CE, PRIMARY)",
            f"- Best epoch: {result_b.best_epoch} (stopped early: {result_b.stopped_early})",
            f"- Recall: {result_b.best_val_recall:.4f}",
            f"- PR-AUC: {result_b.best_val_pr_auc:.4f}",
            f"- F1: {result_b.best_val_f1:.4f}",
            f"- Precision: {result_b.best_val_precision:.4f}",
            "",
            "## Selection rule (locked, design doc section 9)",
            "PRIMARY = defect recall, SECONDARY = defect PR-AUC, TERTIARY = defect F1.",
            "Both experiments reported in full regardless of outcome.",
        ]
        path = self.reports_dir / "phase3_model_selection_report.md"
        path.write_text("\n".join(lines))
        return path
