import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

TOTAL_STEPS = 30
step_counter = 0


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # absolute paths
    config["data"]["raw_dir"] = ROOT / config["data"]["raw_dir"]
    config["data"]["splits_dir"] = ROOT / config["data"]["splits_dir"]
    config["data"]["splits_dir"].mkdir(parents=True, exist_ok=True)

    for name in config["paths"]:
        config["paths"][name] = ROOT / config["paths"][name]
        config["paths"][name].mkdir(parents=True, exist_ok=True)

    return config


def save_to_config(section, key, value):
    # reload
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    raw[section][key] = value

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def relative_path(path):
    return os.path.relpath(path, ROOT)


def full_path(path):
    return ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m {seconds - minutes * 60:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    rest = seconds - hours * 3600 - minutes * 60
    return f"{hours}h {minutes:02d}m {rest:04.1f}s"


def start_step(title):
    global step_counter
    step_counter += 1
    print()
    print("=" * 78)
    print(f"[{step_counter}/{TOTAL_STEPS}] {title}")
    print("=" * 78)
    return time.time()


def end_step(start_time):
    print(f"Step done in {format_time(time.time() - start_time)}")
