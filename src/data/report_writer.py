import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.models import DuplicateReport, GateResult, ImbalanceReport


class Phase1ReportWriter:

    def __init__(self, figures_dir: Path, splits_dir: Path, reports_dir: Path):
        self.figures_dir = figures_dir
        self.splits_dir = splits_dir
        self.reports_dir = reports_dir

    def save_distribution_plot(
        self, report: ImbalanceReport, title: str, filename: str
    ) -> None:
        plt.figure(figsize=(5, 4))
        names = list(report.class_counts.keys())
        counts = list(report.class_counts.values())
        plt.bar(names, counts, color=["#4C72B0", "#C44E52"])
        plt.title(title)
        plt.ylabel("Image count")
        for i, v in enumerate(counts):
            plt.text(i, v + 30, str(v), ha="center")
        plt.savefig(self.figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def save_constructed_pool(self, df: pd.DataFrame) -> Path:
        out = df.copy()
        out["filepath"] = out["filepath"].apply(
            lambda p: os.path.relpath(p, start=".")
        )
        path = self.splits_dir / "constructed_pool.csv"
        out.to_csv(path, index=False)
        return path

    def save_duplicate_groups(self, duplicates: DuplicateReport) -> Path:
        rows = []
        for group_id, group in enumerate(duplicates.groups):
            for row in group:
                rows.append({
                    "duplicate_group_id": group_id,
                    "filepath": os.path.relpath(row["filepath"], start="."),
                    "class_name": row["class_name"],
                    "source_subfolder": row["source_subfolder"],
                })
        path = self.splits_dir / "duplicate_groups.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def write_gate_report(
        self,
        raw_report: ImbalanceReport,
        constructed_report: ImbalanceReport,
        duplicates: DuplicateReport,
        gate_result: GateResult,
    ) -> Path:
        lines = [
            "# Phase 1 Gate Report — Dataset Verification",
            "",
            f"**Verdict: {'PASSED' if gate_result.is_valid else 'FAILED'}.**",
            f"{gate_result.message}",
            "",
            "## Raw pool",
            f"- Total: {raw_report.n_total}",
            f"- {raw_report.majority_class}: {raw_report.n_majority}",
            f"- {raw_report.minority_class}: {raw_report.n_minority}",
            f"- Ratio: {raw_report.imbalance_ratio:.2f}:1",
            "",
            "## Duplicate check",
            f"- Groups found: {duplicates.n_duplicate_groups}",
            "",
            "## Constructed pool",
            f"- Total: {constructed_report.n_total}",
            f"- {constructed_report.majority_class}: {constructed_report.n_majority}",
            f"- {constructed_report.minority_class}: {constructed_report.n_minority}",
            f"- Ratio: {constructed_report.imbalance_ratio:.2f}:1",
            f"- Minority prevalence: {constructed_report.minority_prevalence:.2%}",
        ]
        path = self.reports_dir / "phase1_gate_report.md"
        path.write_text("\n".join(lines))
        return path
