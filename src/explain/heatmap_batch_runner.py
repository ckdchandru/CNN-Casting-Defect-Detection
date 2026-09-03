from pathlib import Path

import pandas as pd
from PIL import Image

from src.explain.gradcam import GradCAM
from src.explain.heatmap_overlay import overlay_heatmap


class HeatmapBatchRunner:

    def __init__(self, gradcam: GradCAM, eval_transform, image_size: int, gradcam_dir: Path):
        self.gradcam = gradcam
        self.eval_transform = eval_transform
        self.image_size = image_size
        self.gradcam_dir = gradcam_dir

    def run(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []

        for _, row in predictions_df.iterrows():
            image = Image.open(row["image_id"]).convert("RGB")
            resized_original = image.resize((self.image_size, self.image_size))
            input_tensor = self.eval_transform(image).unsqueeze(0)

            target_class = int(row["predicted_label"])
            cam = self.gradcam.generate(input_tensor, target_class)
            overlay = overlay_heatmap(resized_original, cam)

            filename = self._build_filename(row)
            path = self.gradcam_dir / filename
            overlay.save(path)

            manifest_rows.append({
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "correct": row["correct"],
                "gradcam_path": str(path),
            })

        return pd.DataFrame(manifest_rows)

    @staticmethod
    def _build_filename(row) -> str:
        stem = Path(row["image_id"]).stem
        correctness = "correct" if row["correct"] else "wrong"
        return f"{stem}__true{row['true_label']}_pred{row['predicted_label']}_{correctness}.png"
