from typing import List

import numpy as np
from sklearn.metrics import precision_score, recall_score

from src.evaluate.models import ThresholdResult, ThresholdRow

DEFECT_LABEL = 1


class ThresholdSelector:

    def __init__(self, precision_floor: float, candidates: List[float]):
        self.precision_floor = precision_floor
        self.candidates = candidates

    def select(self, val_labels: np.ndarray, val_probs: np.ndarray) -> ThresholdResult:
        """Build the table, pick max-recall row above floor."""
        table = self._build_table(val_labels, val_probs)

        eligible = [row for row in table if row.precision >= self.precision_floor]
        if not eligible:
            candidates_str = ", ".join(f"{r.threshold:.2f}" for r in table)
            raise RuntimeError(
                f"No threshold meets precision_floor={self.precision_floor}. "
                f"Checked: {candidates_str}. Reconsider precision_floor or "
                f"the trained model before proceeding."
            )

        chosen = max(eligible, key=lambda row: row.recall)

        return ThresholdResult(
            chosen_threshold=chosen.threshold,
            chosen_precision=chosen.precision,
            chosen_recall=chosen.recall,
            precision_floor=self.precision_floor,
            table=table,
        )

    def _build_table(self, val_labels: np.ndarray, val_probs: np.ndarray) -> List[ThresholdRow]:

        table = []
        for threshold in self.candidates:
            preds = (val_probs >= threshold).astype(int)
            precision = precision_score(
                val_labels, preds, pos_label=DEFECT_LABEL, zero_division=0
            )
            recall = recall_score(
                val_labels, preds, pos_label=DEFECT_LABEL, zero_division=0
            )
            table.append(ThresholdRow(threshold=threshold, precision=precision, recall=recall))
        return table
