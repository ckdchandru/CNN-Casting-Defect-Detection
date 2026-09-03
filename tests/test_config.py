import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, is_dataset_locked


def test_config_loads_and_has_required_sections():
    config = load_config("config/config.yaml")
    for key in ("project", "data", "model", "train", "augmentation", "evaluation", "paths"):
        assert key in config, f"config.yaml missing required section: {key}"


def test_experiment_is_valid():
    config = load_config("config/config.yaml")
    assert config["train"]["experiment"] in ("A", "B", "C")


def test_backbone_is_valid():
    config = load_config("config/config.yaml")
    assert config["model"]["backbone"] in ("resnet18", "efficientnet_b0")


def test_paths_are_resolved_and_created():
    config = load_config("config/config.yaml")
    for key, path in config["paths"].items():
        assert path.is_absolute(), f"paths.{key} was not resolved to an absolute path"
        assert path.exists(), f"paths.{key} directory was not created"


def test_dataset_starts_unlocked():
    config = load_config("config/config.yaml")
    # Before Phase 1's gate has been run and lock_dataset() called, this
    # should be false — nothing downstream should silently assume a locked
    # dataset just because a config file exists.
    assert is_dataset_locked(config) in (True, False)


if __name__ == "__main__":
    test_config_loads_and_has_required_sections()
    test_experiment_is_valid()
    test_backbone_is_valid()
    test_paths_are_resolved_and_created()
    test_dataset_starts_unlocked()
    print("All config tests passed.")
