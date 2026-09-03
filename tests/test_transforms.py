import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms import build_transforms

AUGMENTATION_ON = {
    "horizontal_flip": True,
    "rotation_degrees": 10,
    "brightness_jitter": 0.1,
    "contrast_jitter": 0.1,
}


def test_train_transform_has_augmentation_ops():
    pipelines = build_transforms(image_size=224, augmentation_config=AUGMENTATION_ON)
    train_ops = [type(op).__name__ for op in pipelines["train"].transforms]

    assert "RandomHorizontalFlip" in train_ops
    assert "RandomRotation" in train_ops
    assert "ColorJitter" in train_ops


def test_eval_transform_has_no_augmentation_ops():
    pipelines = build_transforms(image_size=224, augmentation_config=AUGMENTATION_ON)
    eval_ops = [type(op).__name__ for op in pipelines["eval"].transforms]

    for op_name in ("RandomHorizontalFlip", "RandomRotation", "ColorJitter"):
        assert op_name not in eval_ops, f"{op_name} leaked into eval transform"


def test_both_transforms_resize_and_normalize():
    pipelines = build_transforms(image_size=224, augmentation_config=AUGMENTATION_ON)
    for split_name in ("train", "eval"):
        ops = [type(op).__name__ for op in pipelines[split_name].transforms]
        assert "Resize" in ops
        assert "Normalize" in ops


if __name__ == "__main__":
    test_train_transform_has_augmentation_ops()
    test_eval_transform_has_no_augmentation_ops()
    test_both_transforms_resize_and_normalize()
    print("All transform tests passed.")
