from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class EDAReporter:

    def __init__(self, figures_dir: Path):
        self.figures_dir = figures_dir

    def plot_split_distribution(self, merged_df: pd.DataFrame, filename: str) -> Path:
        counts = merged_df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
        counts = counts.reindex(["train", "val", "test"])

        ax = counts.plot(kind="bar", figsize=(6, 4), color=["#4C72B0", "#C44E52"])
        ax.set_title("Class distribution per split")
        ax.set_ylabel("Image count")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = self.figures_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def save_sample_grid(
        self, merged_df: pd.DataFrame, n_per_class: int, filename: str
    ) -> Path:
        classes = sorted(merged_df["class_name"].unique())
        fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2 * n_per_class, 2 * len(classes)))

        for row, class_name in enumerate(classes):
            samples = merged_df[merged_df["class_name"] == class_name].sample(
                n=n_per_class, random_state=42
            )
            for col, (_, sample_row) in enumerate(samples.iterrows()):
                image = Image.open(sample_row["filepath"])
                ax = axes[row, col] if len(classes) > 1 else axes[col]
                ax.imshow(image, cmap="gray")
                ax.axis("off")
                if col == 0:
                    ax.set_title(class_name, loc="left", fontsize=10)

        plt.suptitle("Sample images per class (train split)")
        plt.tight_layout()
        path = self.figures_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        return path
