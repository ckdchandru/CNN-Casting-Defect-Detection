import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.helpers import full_path

HASH_SIZE = 8


def average_hash(filepath):
    img = Image.open(full_path(filepath)).convert("L")
    img = img.resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    return (pixels > pixels.mean()).flatten()


def add_group_ids(pool_df):
    # same class, same hash
    key_to_group = {}
    group_ids = []

    for filepath, class_name in zip(pool_df["filepath"], pool_df["class_name"]):
        key = (class_name, average_hash(filepath).tobytes())
        if key not in key_to_group:
            key_to_group[key] = f"group_{len(key_to_group)}"
        group_ids.append(key_to_group[key])

    grouped = pool_df.copy()
    grouped["group_id"] = group_ids
    return grouped


def split_by_group(grouped_df, train_frac, val_frac, test_frac, seed):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")

    # one row per group
    groups = (
        grouped_df.groupby("group_id")
        .agg(class_name=("class_name", "first"))
        .reset_index()
    )
    group_ids = groups["group_id"].to_numpy(dtype=object)
    classes = groups["class_name"].to_numpy(dtype=object)

    train_ids, rest_ids, _, rest_classes = train_test_split(
        group_ids, classes, train_size=train_frac, stratify=classes, random_state=seed
    )
    val_share = val_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        rest_ids, train_size=val_share, stratify=rest_classes, random_state=seed
    )

    split_of = {}
    for g in train_ids:
        split_of[g] = "train"
    for g in val_ids:
        split_of[g] = "val"
    for g in test_ids:
        split_of[g] = "test"

    result = grouped_df.copy()
    result["split"] = result["group_id"].map(split_of)
    return result


def check_no_leakage(split_df):
    splits_per_group = split_df.groupby("group_id")["split"].nunique()
    leaking = (splits_per_group > 1).sum()
    if leaking > 0:
        raise AssertionError(f"{leaking} group(s) span multiple splits")


def split_counts(split_df):
    return split_df.groupby(["split", "class_name"]).size().to_dict()


def save_splits(split_df, splits_dir):
    paths = {}
    for name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == name]
        path = splits_dir / f"{name}.csv"
        part[["filepath", "class_name"]].to_csv(path, index=False)
        paths[name] = path
    return paths


def write_near_duplicate_report(grouped_df, path):
    sizes = grouped_df.groupby("group_id").size()
    multi = sizes[sizes > 1]

    lines = [
        "# Phase 2 Near-Duplicate Report",
        "",
        "Images are grouped by an 8x8 average hash (exact hash match,",
        "same class only). This catches images that look the same but",
        "are not byte-for-byte copies, which the MD5 check in Phase 1",
        "misses. The split is done on these groups.",
        "",
        f"- Total images: {len(grouped_df):,}",
        f"- Total groups: {grouped_df['group_id'].nunique():,}",
        f"- Groups with >1 image: {len(multi):,}",
        f"- Images inside a multi-image group: {multi.sum():,}",
        f"- Largest group: {sizes.max()} images",
        "",
        "## Per-class effective independent count",
    ]

    groups_per_class = grouped_df.groupby("class_name")["group_id"].nunique()
    images_per_class = grouped_df.groupby("class_name").size()
    for name in groups_per_class.index:
        lines.append(
            f"- {name}: {groups_per_class[name]:,} groups "
            f"(from {images_per_class[name]:,} raw images)"
        )
    path.write_text("\n".join(lines))


def plot_split_distribution(split_df, path):
    counts = split_df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "val", "test"])

    ax = counts.plot(kind="bar", figsize=(6, 4), color=["#4C72B0", "#C44E52"])
    ax.set_title("Class distribution per split")
    ax.set_ylabel("Image count")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_sample_grid(train_df, n_per_class, path):
    classes = sorted(train_df["class_name"].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2 * n_per_class, 2 * len(classes)))

    for r, name in enumerate(classes):
        samples = train_df[train_df["class_name"] == name].sample(n=n_per_class, random_state=42)
        for c, filepath in enumerate(samples["filepath"]):
            ax = axes[r, c]
            ax.imshow(Image.open(full_path(filepath)), cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_title(name, loc="left", fontsize=10)

    plt.suptitle("Sample images per class (train split)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
