import numpy as np
import torch
import torch.nn as nn


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


class GradCAM:

    def __init__(self, model: nn.Module, target_layer_name: str, device: torch.device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

        target_layer = get_module_by_name(model, target_layer_name)
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input_, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        logits = self.model(image_tensor)
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam
