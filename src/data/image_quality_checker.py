from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.models import ImageQualityReport


class ImageQualityChecker:

    def check(self, df: pd.DataFrame) -> ImageQualityReport:
        sizes_seen: Dict[str, int] = {}
        modes_seen: Dict[str, int] = {}
        unreadable: List[str] = []

        for filepath in tqdm(df["filepath"], desc="Checking images"):
            try:
                self._record_properties(filepath, sizes_seen, modes_seen)
            except Exception:
                unreadable.append(filepath)

        return ImageQualityReport(
            n_images_checked=len(df),
            n_unreadable=len(unreadable),
            unreadable_paths=unreadable,
            sizes_seen=sizes_seen,
            modes_seen=modes_seen,
        )

    @staticmethod
    def _record_properties(filepath: str, sizes: dict, modes: dict) -> None:
        with Image.open(filepath) as img:
            img.verify()
        # verify() closes the file handle
        with Image.open(filepath) as img:
            size_key = f"{img.width}x{img.height}"
            sizes[size_key] = sizes.get(size_key, 0) + 1
            modes[img.mode] = modes.get(img.mode, 0) + 1
