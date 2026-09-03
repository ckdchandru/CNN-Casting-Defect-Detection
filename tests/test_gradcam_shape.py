import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import models

from src.explain.gradcam import GradCAM, get_module_by_name

DEVICE = torch.device("cpu")


def _build_toy_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model


def test_get_module_by_name_resolves_correctly():
    model = _build_toy_model()
    resolved = get_module_by_name(model, "layer4.1.conv2")
    assert resolved is model.layer4[1].conv2


def test_activations_and_gradients_have_expected_shape():
    model = _build_toy_model()
    gradcam = GradCAM(model, "layer4.1.conv2", DEVICE)

    image = torch.randn(1, 3, 224, 224)
    gradcam.generate(image, target_class=1)

    assert gradcam.activations.shape == (1, 512, 7, 7)
    assert gradcam.gradients.shape == (1, 512, 7, 7)


def test_cam_shape_matches_feature_map():
    model = _build_toy_model()
    gradcam = GradCAM(model, "layer4.1.conv2", DEVICE)

    image = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate(image, target_class=0)

    assert cam.shape == (7, 7)


def test_cam_values_in_unit_range():
    model = _build_toy_model()
    gradcam = GradCAM(model, "layer4.1.conv2", DEVICE)

    image = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate(image, target_class=1)

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_generate_works_for_both_classes():
    model = _build_toy_model()
    gradcam = GradCAM(model, "layer4.1.conv2", DEVICE)
    image = torch.randn(1, 3, 224, 224)

    cam_normal = gradcam.generate(image, target_class=0)
    cam_defect = gradcam.generate(image, target_class=1)

    assert cam_normal.shape == cam_defect.shape == (7, 7)


if __name__ == "__main__":
    test_get_module_by_name_resolves_correctly()
    test_activations_and_gradients_have_expected_shape()
    test_cam_shape_matches_feature_map()
    test_cam_values_in_unit_range()
    test_generate_works_for_both_classes()
    print("All Grad-CAM shape tests passed.")
