from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ImbalanceReport:

    class_counts: Dict[str, int]
    n_total: int
    majority_class: str
    minority_class: str
    n_majority: int
    n_minority: int
    imbalance_ratio: float
    minority_prevalence: float


@dataclass
class DuplicateReport:

    n_images_checked: int
    n_duplicate_groups: int
    duplicate_group_sizes: List[int] = field(default_factory=list)
    groups: List[List[dict]] = field(default_factory=list)


@dataclass
class ImageQualityReport:

    n_images_checked: int
    n_unreadable: int
    unreadable_paths: List[str]
    sizes_seen: Dict[str, int]
    modes_seen: Dict[str, int]


@dataclass
class GateResult:

    is_valid: bool
    message: str
