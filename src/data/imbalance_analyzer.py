import pandas as pd

from src.data.models import ImbalanceReport


class ImbalanceAnalyzer:

    def analyze(self, df: pd.DataFrame) -> ImbalanceReport:
        counts = df["class_name"].value_counts().to_dict()

        if len(counts) != 2:
            raise ValueError(f"Expected 2 classes, found {list(counts)}")

        majority_class = max(counts, key=counts.get)
        minority_class = min(counts, key=counts.get)
        n_majority = counts[majority_class]
        n_minority = counts[minority_class]
        n_total = n_majority + n_minority

        return ImbalanceReport(
            class_counts=counts,
            n_total=n_total,
            majority_class=majority_class,
            minority_class=minority_class,
            n_majority=n_majority,
            n_minority=n_minority,
            imbalance_ratio=n_majority / n_minority,
            minority_prevalence=n_minority / n_total,
        )
