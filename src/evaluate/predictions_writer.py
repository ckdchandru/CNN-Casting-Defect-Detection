import os
from pathlib import Path

import numpy as np
import pandas as pd


class PredictionsWriter:

    def write(
        self,
        filepaths: list,
        true_labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        threshold: float,
        predictions_dir: Path,
    ) -> Path:
        relative_paths = [os.path.relpath(fp, start=".") for fp in filepaths]

        df = pd.DataFrame({
            "image_id": relative_paths,
            "true_label": true_labels,
            "predicted_label": preds,
            "probability_defect": probs,
            "threshold": threshold,
            "correct": true_labels == preds,
        })

        path = predictions_dir / "test_predictions.csv"
        df.to_csv(path, index=False)
        return path
