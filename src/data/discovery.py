import os
from pathlib import Path
from typing import Dict

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetDiscovery:

    def __init__(self, raw_dir: Path, class_folders: Dict[str, str]):
        self.raw_dir = raw_dir
        self.class_folders = class_folders

    def discover(self) -> pd.DataFrame:
        """Return one row per image found."""
        # fail fast, missing root
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {self.raw_dir}")

        rows = []
        for class_name, folder_name in self.class_folders.items():
            rows.extend(self._collect_class(class_name, folder_name))

        if not rows:
            raise ValueError(f"No images found under {self.raw_dir}")

        df = pd.DataFrame(rows)
        self._assert_no_overlap(df)
        return df

    def _collect_class(self, class_name: str, folder_name: str) -> list:
        matches = [p for p in self.raw_dir.rglob(folder_name) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"No folder '{folder_name}' for class '{class_name}'"
            )

        rows = []
        for class_dir in matches:
            source = self._source_subfolder(class_dir)
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "filepath": str(path),
                        "class_name": class_name,
                        "source_subfolder": source,
                    })
        return rows

    def _source_subfolder(self, class_dir: Path) -> str:
        try:
            return class_dir.relative_to(self.raw_dir).parts[0]
        except (ValueError, IndexError):
            return "."

    @staticmethod
    def _assert_no_overlap(df: pd.DataFrame) -> None:
        n_dupes = df["filepath"].duplicated().sum()
        if n_dupes:
            raise ValueError(f"{n_dupes} filepath(s) matched two classes")
