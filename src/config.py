
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root, two levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "data",
    "model",
    "train",
    "augmentation",
    "evaluation",
    "paths",
]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:

    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Config not found at {full_path}. "
            f"Expected relative to {PROJECT_ROOT}."
        )

    with open(full_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    _resolve_paths(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"config.yaml missing section(s): {missing}")

    experiment = config["train"].get("experiment")
    if experiment not in ("A", "B", "C"):
        raise ValueError(f"train.experiment must be A/B/C, got {experiment!r}")

    backbone = config["model"].get("backbone")
    if backbone not in ("resnet18", "efficientnet_b0"):
        raise ValueError(f"model.backbone invalid: {backbone!r}")


def _resolve_paths(config: Dict[str, Any]) -> None:

    config["data"]["raw_dir"] = PROJECT_ROOT / config["data"]["raw_dir"]
    config["data"]["splits_dir"] = PROJECT_ROOT / config["data"]["splits_dir"]

    for key, value in config["paths"].items():
        config["paths"][key] = PROJECT_ROOT / value
        config["paths"][key].mkdir(parents=True, exist_ok=True)  # ensure it exists


def is_dataset_locked(config: Dict[str, Any]) -> bool:

    return bool(config["data"].get("locked", False))
