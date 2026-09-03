import matplotlib.cm as cm
import numpy as np
from PIL import Image


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    cam_image = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_image.resize(original_image.size, Image.BILINEAR)
    cam_array = np.asarray(cam_resized) / 255.0

    heatmap_rgba = cm.jet(cam_array)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb).convert("RGB")

    original_rgb = original_image.convert("RGB")
    return Image.blend(original_rgb, heatmap_image, alpha=alpha)
