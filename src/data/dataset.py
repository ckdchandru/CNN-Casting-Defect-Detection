import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.helpers import full_path

# defect = positive class
LABELS = {"normal": 0, "defect": 1}
NORMAL = 0
DEFECT = 1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(image_size, aug):
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    steps = [transforms.Resize((image_size, image_size))]
    if aug.get("horizontal_flip"):
        steps.append(transforms.RandomHorizontalFlip())
    if aug.get("rotation_degrees", 0):
        steps.append(transforms.RandomRotation(aug["rotation_degrees"]))

    brightness = aug.get("brightness_jitter", 0)
    contrast = aug.get("contrast_jitter", 0)
    if brightness or contrast:
        steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(MEAN, STD))
    train_tf = transforms.Compose(steps)

    return train_tf, eval_tf


class CastingDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(full_path(row["filepath"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, LABELS[row["class_name"]]
