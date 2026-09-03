
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Defect is positive class.
# Primary recall target here.
CLASS_TO_LABEL = {"normal": 0, "defect": 1}


class CastingDataset(Dataset):

    def __init__(self, split_csv_path: Path, transform=None):
        self.df = pd.read_csv(split_csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_TO_LABEL[row["class_name"]]
        return image, label
