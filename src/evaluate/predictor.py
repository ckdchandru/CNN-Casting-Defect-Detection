import numpy as np
import torch
from torch.utils.data import DataLoader

DEFECT_LABEL = 1


class Predictor:

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict(self, loader: DataLoader) -> tuple:
        self.model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)

                all_labels.extend(labels.tolist())
                all_probs.extend(probs[:, DEFECT_LABEL].cpu().tolist())

        return np.array(all_labels), np.array(all_probs)
