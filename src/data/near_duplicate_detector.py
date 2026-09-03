import numpy as np
import pandas as pd
from PIL import Image


class NearDuplicateDetector:

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def detect(self, pool_df: pd.DataFrame) -> pd.DataFrame:
        hashes = self._compute_hashes(pool_df)
        group_ids = self._group_same_class_exact_matches(pool_df, hashes)

        result = pool_df.copy()
        result["group_id"] = group_ids
        return result

    def _compute_hashes(self, pool_df: pd.DataFrame) -> np.ndarray:
        n = len(pool_df)
        hashes = np.zeros((n, self.hash_size ** 2), dtype=bool)
        for i, filepath in enumerate(pool_df["filepath"]):
            img = Image.open(filepath).convert("L").resize(
                (self.hash_size, self.hash_size), Image.LANCZOS
            )
            pixels = np.asarray(img, dtype=np.float64)
            hashes[i] = (pixels > pixels.mean()).flatten()
        return hashes

    def _group_same_class_exact_matches(self, pool_df: pd.DataFrame, hashes: np.ndarray) -> list:
        classes = pool_df["class_name"].values
        key_to_group_id = {}
        group_ids = []

        for i in range(len(pool_df)):
            key = (classes[i], hashes[i].tobytes())
            if key not in key_to_group_id:
                key_to_group_id[key] = f"group_{len(key_to_group_id)}"
            group_ids.append(key_to_group_id[key])

        return group_ids
