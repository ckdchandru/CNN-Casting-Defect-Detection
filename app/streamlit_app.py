import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
from PIL import Image

from src.config import load_config
from src.data.transforms import build_transforms
from src.model.backbone import build_backbone
from src.explain.gradcam import GradCAM
from src.explain.heatmap_overlay import overlay_heatmap

DEFECT_LABEL = 1
NORMAL_LABEL = 0


@st.cache_resource
def load_locked_pipeline():
    """Config, model, device - loaded once per session."""
    config = load_config("config/config.yaml")
    winner = config["train"].get("locked_winner")
    if not winner:
        st.error(
            "No locked model found. Run `python main.py` first "
            "(needs Phases 1-3 complete)."
        )
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(
        name=config["model"]["backbone"],
        pretrained=False,
        num_classes=config["model"]["num_classes"],
    ).to(device)
    checkpoint_path = config["paths"]["checkpoints_dir"] / f"experiment_{winner}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    gradcam = GradCAM(model, config["model"]["gradcam_target_layer"], device)
    transform_pipelines = build_transforms(
        image_size=config["train"]["image_size"],
        augmentation_config=config["augmentation"],
    )

    return config, model, device, winner, gradcam, transform_pipelines["eval"]


def predict(model, device, eval_transform, image: Image.Image, image_size: int):
    """P(defect) for one image."""
    input_tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
    return input_tensor, probs[0, DEFECT_LABEL].item()


def main():
    st.set_page_config(page_title="Casting Defect Detection", layout="centered")

    config, model, device, winner, gradcam, eval_transform = load_locked_pipeline()
    threshold = config["evaluation"].get("locked_threshold")
    if threshold is None:
        st.warning(
            "No locked threshold found - using 0.5 as a placeholder. "
            "Run `python main.py` (Phase 4) for the real tuned value."
        )
        threshold = 0.5

    st.title("Casting Defect Detection")
    st.caption(f"Locked model: Experiment {winner}  |  Decision threshold: {threshold:.2f}")

    uploaded_file = st.file_uploader(
        "Upload a top-view casting image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_size = config["train"]["image_size"]

    input_tensor, probability_defect = predict(model, device, eval_transform, image, image_size)
    predicted_label = DEFECT_LABEL if probability_defect >= threshold else NORMAL_LABEL
    prediction_text = "DEFECT" if predicted_label == DEFECT_LABEL else "NORMAL"

    cam = gradcam.generate(input_tensor, target_class=predicted_label)
    resized_original = image.resize((image_size, image_size))
    overlay = overlay_heatmap(resized_original, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_original, caption="Original", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    st.metric("Prediction", prediction_text)
    st.metric("P(defect)", f"{probability_defect:.3f}")

    st.caption(
        "Grad-CAM shows the spatial region that most influenced this "
        "prediction. It is a qualitative spatial-alignment check, not "
        "proof of causal reasoning, and not a quantitative localization "
        "metric - no pixel-level ground-truth masks exist for this dataset."
    )


if __name__ == "__main__":
    main()
