from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils.helpers import full_path, relative_path


def get_layer(model, name):
    layer = model
    for part in name.split("."):
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer


class GradCAM:

    def __init__(self, model, layer_name, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

        layer = get_layer(model, layer_name)
        layer.register_forward_hook(self.save_activations)
        layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class):
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        outputs = self.model(image_tensor)
        self.model.zero_grad()
        outputs[0, target_class].backward()

        # average gradient per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        low, high = cam.min(), cam.max()
        if high - low > 1e-8:
            cam = (cam - low) / (high - low)
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_heatmap(image, cam, alpha=0.4):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize(image.size, Image.BILINEAR)
    cam_array = np.asarray(cam_img) / 255.0

    colored = plt.cm.jet(cam_array)[:, :, :3]
    heatmap = Image.fromarray((colored * 255).astype(np.uint8)).convert("RGB")
    return Image.blend(image.convert("RGB"), heatmap, alpha=alpha)


def make_all_heatmaps(gradcam, predictions_df, eval_tf, image_size, out_dir):
    rows = []
    for _, row in predictions_df.iterrows():
        image = Image.open(full_path(row["image_id"])).convert("RGB")
        small = image.resize((image_size, image_size))
        tensor = eval_tf(image).unsqueeze(0)

        cam = gradcam.generate(tensor, int(row["predicted_label"]))
        overlay = overlay_heatmap(small, cam)

        status = "correct" if row["correct"] else "wrong"
        # same names in train/test
        folder = Path(row["image_id"]).parent.parent.name
        name = f"{folder}_{Path(row['image_id']).stem}__true{row['true_label']}_pred{row['predicted_label']}_{status}.png"
        save_path = out_dir / name
        overlay.save(save_path)

        rows.append({
            "image_id": row["image_id"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "correct": row["correct"],
            "gradcam_path": relative_path(save_path),
        })

    return pd.DataFrame(rows)


def true_positive_grid(manifest_df, n_samples, path):
    tp = manifest_df[(manifest_df["true_label"] == 1) & (manifest_df["predicted_label"] == 1)]
    n_samples = min(n_samples, len(tp))
    samples = tp.sample(n=n_samples, random_state=42)

    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")
    for ax, img_path in zip(axes, samples["gradcam_path"]):
        ax.imshow(Image.open(full_path(img_path)))

    plt.suptitle("True positives - Grad-CAM overlay (gate check)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_phase5_report(manifest_df, grid_path, verdict, notes, path):
    n_tp = len(manifest_df[(manifest_df["true_label"] == 1) & (manifest_df["predicted_label"] == 1)])
    lines = [
        "# Phase 5 Grad-CAM Report",
        "",
        f"- Heatmaps generated: {len(manifest_df)} (every test prediction)",
        f"- Correct predictions: {manifest_df['correct'].sum()}",
        f"- True positives available for gate review: {n_tp}",
        f"- Sample grid for gate review: {grid_path}",
        "",
        "## What Grad-CAM does and does not prove",
        "Grad-CAM shows the spatial regions that contributed most to the",
        "model's activation for the predicted class. It does NOT prove",
        "causality, and does NOT prove the model is \"looking at the",
        "defect\" in any certified sense.",
        "",
        "No pixel-level ground-truth defect masks exist for this dataset,",
        "so the gate check is a QUALITATIVE spatial-alignment sanity",
        "check on a sample of true positives, not a quantitative",
        "localization-accuracy metric.",
        "",
        "## Gate verdict (human review)",
        f"**{verdict}.** {notes}",
    ]
    path.write_text("\n".join(lines))
