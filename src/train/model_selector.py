from dataclasses import dataclass
from src.train.models import FitResult


@dataclass
class SelectionResult:

    winner: str
    reason: str


class ModelSelector:

    def select(self, result_a: FitResult, result_b: FitResult) -> SelectionResult:
        """Pick A or B by the exact locked tie-break."""
        metrics_a = self._best_metrics(result_a)
        metrics_b = self._best_metrics(result_b)

        for metric_name in ("recall", "pr_auc", "f1"):
            value_a = metrics_a[metric_name]
            value_b = metrics_b[metric_name]
            if value_a != value_b:
                winner = "A" if value_a > value_b else "B"
                reason = f"{metric_name}: A={value_a:.4f} vs B={value_b:.4f}"
                return SelectionResult(winner=winner, reason=reason)

        return SelectionResult(winner="A", reason="exact tie on all three metrics")

    @staticmethod
    def _best_metrics(result: FitResult) -> dict:
        return {
            "recall": result.best_val_recall,
            "pr_auc": result.best_val_pr_auc,
            "f1": result.best_val_f1,
        }
