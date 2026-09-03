from pathlib import Path
from typing import Any

import yaml


def update_config_field(config_path: Path, section: str, key: str, value: Any) -> None:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    raw_config[section][key] = value

    with open(config_path, "w") as f:
        yaml.safe_dump(raw_config, f, sort_keys=False)
