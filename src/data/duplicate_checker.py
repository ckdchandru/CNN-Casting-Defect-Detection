import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.data.models import DuplicateReport


class DuplicateChecker:

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size

    def check(self, df: pd.DataFrame) -> DuplicateReport:
        """Hash every row, group matches."""
        hash_to_rows: Dict[str, List[dict]] = {}

        # hash every file
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Hashing images"):
            digest = self._hash_file(Path(row["filepath"]))
            hash_to_rows.setdefault(digest, []).append(row.to_dict())

        groups = [rows for rows in hash_to_rows.values() if len(rows) > 1]

        return DuplicateReport(
            n_images_checked=len(df),
            n_duplicate_groups=len(groups),
            duplicate_group_sizes=[len(g) for g in groups],
            groups=groups,
        )

    def _hash_file(self, path: Path) -> str:
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
