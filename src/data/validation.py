import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.helpers import relative_path

IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

MIN_RATIO = 2.0
MIN_MINORITY = 200


def find_images(raw_dir, class_folders):
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    rows = []
    for class_name, folder_name in class_folders.items():
        folders = [p for p in raw_dir.rglob(folder_name) if p.is_dir()]
        if len(folders) == 0:
            raise FileNotFoundError(f"No folder '{folder_name}' for class '{class_name}'")

        for folder in folders:
            # subfolder
            source = folder.relative_to(raw_dir).parts[0]
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_TYPES:
                    rows.append({
                        "filepath": str(path),
                        "class_name": class_name,
                        "source_subfolder": source,
                    })

    if len(rows) == 0:
        raise ValueError(f"No images found under {raw_dir}")

    df = pd.DataFrame(rows)
    if df["filepath"].duplicated().sum() > 0:
        raise ValueError("Some files matched two classes")
    return df


def class_summary(df):
    counts = df["class_name"].value_counts().to_dict()
    if len(counts) != 2:
        raise ValueError(f"Expected 2 classes, found {list(counts)}")

    majority = max(counts, key=counts.get)
    minority = min(counts, key=counts.get)
    total = counts[majority] + counts[minority]

    return {
        "counts": counts,
        "total": total,
        "majority": majority,
        "minority": minority,
        "n_majority": counts[majority],
        "n_minority": counts[minority],
        "ratio": counts[majority] / counts[minority],
        "prevalence": counts[minority] / total,
    }


def md5_of_file(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def find_exact_duplicates(df):
    hash_to_rows = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Hashing images"):
        digest = md5_of_file(row["filepath"])
        if digest not in hash_to_rows:
            hash_to_rows[digest] = []
        hash_to_rows[digest].append(row.to_dict())

    groups = []
    for rows in hash_to_rows.values():
        if len(rows) > 1:
            groups.append(rows)
    return groups


def check_images(df):
    sizes = {}
    modes = {}
    broken = []

    for filepath in tqdm(df["filepath"], desc="Checking images"):
        try:
            with Image.open(filepath) as img:
                img.verify()
            # verify closes file
            with Image.open(filepath) as img:
                size = f"{img.width}x{img.height}"
                sizes[size] = sizes.get(size, 0) + 1
                modes[img.mode] = modes.get(img.mode, 0) + 1
        except Exception:
            broken.append(filepath)

    return {"sizes": sizes, "modes": modes, "broken": broken}


def make_imbalanced_pool(df, majority_class, minority_class, target_ratio, seed):
    majority_df = df[df["class_name"] == majority_class]
    minority_df = df[df["class_name"] == minority_class]

    keep_n = round(len(majority_df) / target_ratio)
    if keep_n >= len(minority_df):
        raise ValueError(f"target_ratio too low for {len(minority_df)} images")

    sampled = minority_df.sample(n=keep_n, random_state=seed)
    return pd.concat([majority_df, sampled], ignore_index=True)


def check_gate(summary):
    problems = []
    if summary["ratio"] < MIN_RATIO:
        problems.append(f"ratio {summary['ratio']:.2f} below {MIN_RATIO}")
    if summary["n_minority"] < MIN_MINORITY:
        problems.append(f"minority count {summary['n_minority']} below {MIN_MINORITY}")

    if problems:
        return False, "; ".join(problems)
    return True, f"ratio {summary['ratio']:.2f}:1, {summary['n_minority']} minority samples"


def plot_class_counts(summary, title, path):
    names = list(summary["counts"].keys())
    values = list(summary["counts"].values())

    plt.figure(figsize=(5, 4))
    plt.bar(names, values, color=["#4C72B0", "#C44E52"])
    plt.title(title)
    plt.ylabel("Image count")
    for i, v in enumerate(values):
        plt.text(i, v + 30, str(v), ha="center")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_pool(df, path):
    out = df.copy()
    out["filepath"] = out["filepath"].apply(relative_path)
    out.to_csv(path, index=False)


def save_duplicate_groups(groups, path):
    rows = []
    for group_id, group in enumerate(groups):
        for item in group:
            rows.append({
                "duplicate_group_id": group_id,
                "filepath": relative_path(item["filepath"]),
                "class_name": item["class_name"],
                "source_subfolder": item["source_subfolder"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_phase1_report(raw, pool, n_dup_groups, passed, message, path):
    verdict = "PASSED" if passed else "FAILED"
    lines = [
        "# Phase 1 Gate Report - Dataset Verification",
        "",
        f"**Verdict: {verdict}.**",
        message,
        "",
        "## Raw pool",
        f"- Total: {raw['total']}",
        f"- {raw['majority']}: {raw['n_majority']}",
        f"- {raw['minority']}: {raw['n_minority']}",
        f"- Ratio: {raw['ratio']:.2f}:1",
        "",
        "## Duplicate check",
        f"- Groups found: {n_dup_groups}",
        "",
        "## Constructed pool",
        f"- Total: {pool['total']}",
        f"- {pool['majority']}: {pool['n_majority']}",
        f"- {pool['minority']}: {pool['n_minority']}",
        f"- Ratio: {pool['ratio']:.2f}:1",
        f"- Minority prevalence: {pool['prevalence']:.2%}",
    ]
    path.write_text("\n".join(lines))
