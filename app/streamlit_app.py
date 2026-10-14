import sys
from pathlib import Path

# project root import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
from PIL import Image

from src.data.dataset import DEFECT, NORMAL, get_transforms
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.resnet import load_model
from src.utils.helpers import get_device, load_config


@st.cache_resource
def load_everything():
    config = load_config()
    winner = config["train"]["winner"]
    if not winner:
        st.error("No trained model yet. Run `python main.py` first.")
        st.stop()

    device = get_device()
    model, _ = load_model(config, winner, device)
    model.eval()

    cam = GradCAM(model, config["model"]["gradcam_layer"], device)
    _, eval_tf = get_transforms(config["train"]["image_size"], config["augmentation"])
    return config, model, device, winner, cam, eval_tf


def main():
    st.set_page_config(page_title="Casting Defect Detection", layout="centered")
    config, model, device, winner, cam, eval_tf = load_everything()

    threshold = config["evaluation"]["threshold"]
    if threshold is None:
        st.warning("No tuned threshold found, using 0.5.")
        threshold = 0.5

    st.title("Casting Defect Detection")
    st.caption(f"Model: Experiment {winner}  |  Threshold: {threshold:.2f}")

    uploaded = st.file_uploader("Upload a top-view casting image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_size = config["train"]["image_size"]

    tensor = eval_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_defect = torch.softmax(model(tensor), dim=1)[0, DEFECT].item()

    label = DEFECT if prob_defect >= threshold else NORMAL
    heatmap = cam.generate(tensor, label)
    small = image.resize((image_size, image_size))
    overlay = overlay_heatmap(small, heatmap)

    left, right = st.columns(2)
    left.image(small, caption="Original")
    right.image(overlay, caption="Grad-CAM overlay")

    st.metric("Prediction", "DEFECT" if label == DEFECT else "NORMAL")
    st.metric("P(defect)", f"{prob_defect:.3f}")
    st.caption(
        "Grad-CAM shows which region most influenced this prediction. "
        "It is a qualitative check only, not proof of causality - "
        "no pixel-level defect masks exist for this dataset."
    )


main()
