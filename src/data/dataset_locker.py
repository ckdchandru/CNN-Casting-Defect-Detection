from pathlib import Path

import yaml

from src.data.models import ImbalanceReport


class DatasetLocker:

    def lock(
        self, config_path: Path, report: ImbalanceReport
    ) -> None:
        """Write locked, ratio, prevalence to YAML."""
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        raw_config["data"]["locked"] = True
        raw_config["data"]["imbalance_ratio"] = round(report.imbalance_ratio, 4)
        raw_config["data"]["minority_prevalence"] = round(
            report.minority_prevalence, 4
        )

        with open(config_path, "w") as f:
            yaml.safe_dump(raw_config, f, sort_keys=False)
