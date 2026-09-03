from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    train_csv_path: Path, class_to_label: Dict[str, int]
) -> WeightedRandomSampler:
    train_df = pd.read_csv(train_csv_path)
    counts = train_df["class_name"].value_counts()

    class_weight = {
        class_name: 1.0 / count for class_name, count in counts.items()
    }
    sample_weights = train_df["class_name"].map(class_weight).values

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
    )
