from typing import Dict

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, augmentation_config: dict) -> Dict[str, transforms.Compose]:
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ops = [transforms.Resize((image_size, image_size))]

    if augmentation_config.get("horizontal_flip"):
        train_ops.append(transforms.RandomHorizontalFlip())

    rotation_degrees = augmentation_config.get("rotation_degrees", 0)
    if rotation_degrees:
        train_ops.append(transforms.RandomRotation(rotation_degrees))

    brightness = augmentation_config.get("brightness_jitter", 0)
    contrast = augmentation_config.get("contrast_jitter", 0)
    if brightness or contrast:
        train_ops.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_transform = transforms.Compose(train_ops)

    return {"train": train_transform, "eval": eval_transform}
