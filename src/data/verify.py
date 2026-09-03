import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImbalanceReport:

    class_counts: Dict[str, int]
    n_total: int
    majority_class: str
    minority_class: str
    n_majority: int
    n_minority: int
    imbalance_ratio: float          # N_majority / N_minority
    minority_prevalence: float      # N_minority / N_total


@dataclass
class DuplicateReport:

    n_images_checked: int
    n_duplicate_groups: int
    duplicate_group_sizes: List[int] = field(default_factory=list)
    example_groups: List[List[str]] = field(default_factory=list)  # first few, for review


@dataclass
class ImagePropertyReport:

    n_images_checked: int
    n_unreadable: int
    unreadable_paths: List[str]
    sizes_seen: Dict[str, int]      # "WIDTHxHEIGHT" -> count
    modes_seen: Dict[str, int]      # "RGB"/"L"/etc -> count


def list_labeled_images(raw_dir: Path, class_folders: Dict[str, str]) -> pd.DataFrame:

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir {raw_dir} does not exist. Unzip the downloaded dataset "
            f"there first, then confirm the class folder names in "
            f"config.yaml match what's actually inside it."
        )

    rows = []
    for class_name, folder_name in class_folders.items():
        matching_dirs = [
            p for p in raw_dir.rglob(folder_name) if p.is_dir()
        ]
        if not matching_dirs:
            raise FileNotFoundError(
                f"No folder named '{folder_name}' for class '{class_name}' "
                f"found anywhere under {raw_dir}. List {raw_dir} and fix "
                f"config.data.class_folders to match the real folder names "
                f"- don't guess."
            )
        for class_dir in matching_dirs:
            # Record which top-level original subfolder (e.g. train/test)
            # this came from, purely for traceability - not used downstream.
            try:
                source_subfolder = class_dir.relative_to(raw_dir).parts[0]
            except (ValueError, IndexError):
                source_subfolder = "."
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "filepath": str(path),
                        "class_name": class_name,
                        "source_subfolder": source_subfolder,
                    })

    if not rows:
        raise ValueError(
            f"No images with extensions {IMAGE_EXTENSIONS} found under "
            f"{raw_dir}. Check the dataset actually unzipped correctly."
        )

    df = pd.DataFrame(rows)
    duplicate_paths = df["filepath"].duplicated().sum()
    if duplicate_paths:
        raise ValueError(
            f"{duplicate_paths} filepath(s) matched more than one class "
            f"folder pattern - check class_folders in config.yaml for "
            f"overlapping/nested names."
        )
    return df


def compute_imbalance_report(df: pd.DataFrame) -> ImbalanceReport:
    
    counts = df["class_name"].value_counts().to_dict()
    if len(counts) != 2:
        raise ValueError(
            f"Expected exactly 2 classes for this binary gate, found "
            f"{len(counts)}: {list(counts.keys())}. If you're using a "
            f"multi-class dataset (e.g. NEU), this function needs adapting."
        )

    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    n_majority = counts[majority_class]
    n_minority = counts[minority_class]
    n_total = n_majority + n_minority

    return ImbalanceReport(
        class_counts=counts,
        n_total=n_total,
        majority_class=majority_class,
        minority_class=minority_class,
        n_majority=n_majority,
        n_minority=n_minority,
        imbalance_ratio=n_majority / n_minority,
        minority_prevalence=n_minority / n_total,
    )


def _file_md5(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_exact_duplicates(df: pd.DataFrame, max_examples: int = 5) -> DuplicateReport:
    
    hash_to_paths: Dict[str, List[str]] = {}
    for filepath in df["filepath"]:
        digest = _file_md5(Path(filepath))
        hash_to_paths.setdefault(digest, []).append(filepath)

    duplicate_groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]

    return DuplicateReport(
        n_images_checked=len(df),
        n_duplicate_groups=len(duplicate_groups),
        duplicate_group_sizes=[len(g) for g in duplicate_groups],
        example_groups=duplicate_groups[:max_examples],
    )


def check_image_properties(df: pd.DataFrame) -> ImagePropertyReport:
    
    sizes_seen: Dict[str, int] = {}
    modes_seen: Dict[str, int] = {}
    unreadable: List[str] = []

    for filepath in df["filepath"]:
        try:
            with Image.open(filepath) as img:
                img.verify()
            # Re-open after verify() - verify() leaves the file unusable for further reads
            with Image.open(filepath) as img:
                size_key = f"{img.width}x{img.height}"
                sizes_seen[size_key] = sizes_seen.get(size_key, 0) + 1
                modes_seen[img.mode] = modes_seen.get(img.mode, 0) + 1
        except Exception:
            unreadable.append(filepath)

    return ImagePropertyReport(
        n_images_checked=len(df),
        n_unreadable=len(unreadable),
        unreadable_paths=unreadable,
        sizes_seen=sizes_seen,
        modes_seen=modes_seen,
    )


def construct_imbalance(
    df: pd.DataFrame,
    majority_class: str,
    minority_class: str,
    target_ratio: float,
    seed: int,
) -> pd.DataFrame:
    
    majority_df = df[df["class_name"] == majority_class]
    minority_df_full = df[df["class_name"] == minority_class]

    n_majority = len(majority_df)
    target_n_minority = round(n_majority / target_ratio)

    if target_n_minority >= len(minority_df_full):
        raise ValueError(
            f"target_ratio={target_ratio} implies keeping {target_n_minority} "
            f"'{minority_class}' images, but only {len(minority_df_full)} exist. "
            f"Lower target_ratio or re-check which class is which."
        )

    minority_df_sampled = minority_df_full.sample(
        n=target_n_minority, random_state=seed
    )

    constructed = pd.concat([majority_df, minority_df_sampled], ignore_index=True)

    print(
        f"Constructed imbalance: kept all {n_majority} '{majority_class}' images, "
        f"randomly sampled {target_n_minority} of {len(minority_df_full)} "
        f"'{minority_class}' images (seed={seed}) -> "
        f"ratio {n_majority / target_n_minority:.2f}:1"
    )
    return constructed


def print_gate_summary(
    imbalance: ImbalanceReport,
    duplicates: DuplicateReport,
    image_props: ImagePropertyReport,
) -> None:
   
    print("=" * 70)
    print("PHASE 1 GATE — DATASET VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"\nClass counts:            {imbalance.class_counts}")
    print(f"Total images:             {imbalance.n_total}")
    print(f"Majority class:           {imbalance.majority_class} (n={imbalance.n_majority})")
    print(f"Minority class:           {imbalance.minority_class} (n={imbalance.n_minority})")
    print(f"Imbalance ratio (maj/min):{imbalance.imbalance_ratio:.2f}")
    print(f"Minority prevalence:      {imbalance.minority_prevalence:.2%}")

    print(f"\nExact-duplicate groups found: {duplicates.n_duplicate_groups}")
    if duplicates.n_duplicate_groups:
        print(f"  Group sizes: {duplicates.duplicate_group_sizes}")
        print("  These images must not be split across train/val/test independently -")
        print("  either dedupe or split by duplicate-group id in Phase 2.")

    print(f"\nImages checked for readability: {image_props.n_images_checked}")
    print(f"Unreadable/corrupt images:      {image_props.n_unreadable}")
    if image_props.unreadable_paths:
        print(f"  e.g. {image_props.unreadable_paths[:5]}")
    print(f"Distinct image sizes seen:  {len(image_props.sizes_seen)}  -> {image_props.sizes_seen}")
    print(f"Distinct image modes seen:  {image_props.modes_seen}")

    print("\n" + "-" * 70)
    print("GATE CHECKLIST (judge against the real numbers above, per design doc):")
    print("  (i)  Is the minority class MATERIALLY smaller than the majority?")
    print("  (ii) Does the minority class have ENOUGH samples for a real")
    print("       train/val/test split (not just a handful of images)?")
    print("If YES to both -> call lock_dataset(config) to flip data.locked=true.")
    print("If NO to either -> switch dataset or build a deliberate imbalanced")
    print("subsample BEFORE writing any training code.")
    print("=" * 70)


def lock_dataset(config: dict, imbalance: ImbalanceReport, config_path: str) -> None:
   
    import yaml

    config["data"]["locked"] = True
    config["data"]["imbalance_ratio"] = round(imbalance.imbalance_ratio, 4)
    config["data"]["minority_prevalence"] = round(imbalance.minority_prevalence, 4)

    # Write back using plain (non-resolved) paths - re-read raw yaml to avoid
    # persisting the absolute-path resolution that src/config.py applies in memory.
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    raw_config["data"]["locked"] = True
    raw_config["data"]["imbalance_ratio"] = round(imbalance.imbalance_ratio, 4)
    raw_config["data"]["minority_prevalence"] = round(imbalance.minority_prevalence, 4)

    with open(config_path, "w") as f:
        yaml.safe_dump(raw_config, f, sort_keys=False)

    print(f"\nDataset LOCKED. imbalance_ratio={imbalance.imbalance_ratio:.2f}, "
          f"minority_prevalence={imbalance.minority_prevalence:.2%} written to {config_path}.")
