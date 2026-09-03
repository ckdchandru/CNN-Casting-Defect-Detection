from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluate.models import EvaluationMetrics, ThresholdResult


class Phase4ReportWriter:

    def __init__(self, figures_dir: Path, reports_dir: Path):
        self.figures_dir = figures_dir
        self.reports_dir = reports_dir

    def save_pr_curve_plot(self, threshold_result: ThresholdResult) -> Path:
        thresholds = [row.threshold for row in threshold_result.table]
        precisions = [row.precision for row in threshold_result.table]
        recalls = [row.recall for row in threshold_result.table]

        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, precisions, label="precision", marker="o")
        plt.plot(thresholds, recalls, label="recall", marker="o")
        plt.axhline(threshold_result.precision_floor, color="gray", linestyle="--", label="P_min")
        plt.axvline(threshold_result.chosen_threshold, color="red", linestyle=":", label="chosen")
        plt.title("Val precision/recall vs threshold")
        plt.xlabel("Threshold")
        plt.legend()
        plt.tight_layout()

        path = self.figures_dir / "phase4_threshold_curve.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def save_confusion_matrix_plot(self, metrics: EvaluationMetrics, filename: str) -> Path:
        matrix = metrics.confusion_matrix

        plt.figure(figsize=(4, 4))
        plt.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(matrix[i][j]), ha="center", va="center")
        plt.xticks([0, 1], ["normal", "defect"])
        plt.yticks([0, 1], ["normal", "defect"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Test confusion matrix")
        plt.tight_layout()

        path = self.figures_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def write_report(
        self,
        winner: str,
        threshold_result: ThresholdResult,
        test_metrics: EvaluationMetrics,
        baseline_metrics: EvaluationMetrics,
        predictions_path: Path,
    ) -> Path:
        lines = [
            "# Phase 4 Evaluation Report",
            "",
            f"**Locked model: Experiment {winner}.**",
            "",
            "## Threshold selection (VALIDATION only)",
            f"Rule: maximize defect recall subject to defect precision >= "
            f"{threshold_result.precision_floor} (stated before inspecting the table).",
            "",
            "| threshold | precision | recall |",
            "|---|---|---|",
        ]
        for row in threshold_result.table:
            lines.append(f"| {row.threshold:.2f} | {row.precision:.4f} | {row.recall:.4f} |")

        lines += [
            "",
            f"**Chosen threshold: {threshold_result.chosen_threshold:.2f}** "
            f"(precision={threshold_result.chosen_precision:.4f}, "
            f"recall={threshold_result.chosen_recall:.4f})",
            "",
            "## Test set (touched once, this evaluation)",
            f"- Precision: {test_metrics.precision:.4f}",
            f"- Recall: {test_metrics.recall:.4f}",
            f"- F1: {test_metrics.f1:.4f}",
            f"- PR-AUC: {test_metrics.pr_auc:.4f}",
            f"- Accuracy (secondary): {test_metrics.accuracy:.4f}",
            f"- Confusion matrix: {test_metrics.confusion_matrix}",
            "",
            "## Majority-class do-nothing baseline (test set)",
            f"- Precision: {baseline_metrics.precision:.4f}",
            f"- Recall: {baseline_metrics.recall:.4f}",
            f"- F1: {baseline_metrics.f1:.4f}",
            f"- PR-AUC: {baseline_metrics.pr_auc:.4f}",
            f"- Accuracy: {baseline_metrics.accuracy:.4f}",
            "",
            "## Frozen predictions",
            f"Saved to: {predictions_path}",
            "Phase 5/6 read this file post-hoc and must not alter it.",
        ]

        path = self.reports_dir / "phase4_evaluation_report.md"
        path.write_text("\n".join(lines))
        return path
