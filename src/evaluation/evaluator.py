import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.data.dataset import DEFECT, NORMAL


def predict(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_labels += labels.tolist()
            all_probs += probs[:, DEFECT].cpu().tolist()

    return np.array(all_labels), np.array(all_probs)


def threshold_table(labels, probs, candidates):
    table = []
    for t in candidates:
        preds = (probs >= t).astype(int)
        table.append({
            "threshold": t,
            "precision": precision_score(labels, preds, pos_label=DEFECT, zero_division=0),
            "recall": recall_score(labels, preds, pos_label=DEFECT, zero_division=0),
        })
    return table


def choose_threshold(table, precision_floor):
    allowed = [row for row in table if row["precision"] >= precision_floor]
    if len(allowed) == 0:
        raise RuntimeError(f"No threshold meets precision_floor={precision_floor}")

    # highest recall wins
    return max(allowed, key=lambda row: row["recall"])


def get_metrics(labels, preds, probs):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[DEFECT], average=None, zero_division=0
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "pr_auc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[NORMAL, DEFECT]).tolist(),
    }


def baseline_metrics(labels):
    # model that always predicts normal
    preds = np.full_like(labels, NORMAL)
    probs = np.zeros(len(labels))
    return get_metrics(labels, preds, probs)


def save_predictions(image_ids, labels, preds, probs, threshold, path):
    df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": labels,
        "predicted_label": preds,
        "probability_defect": probs,
        "threshold": threshold,
        "correct": labels == preds,
    })
    df.to_csv(path, index=False)


def plot_threshold_curve(table, chosen, precision_floor, path):
    thresholds = [row["threshold"] for row in table]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, [row["precision"] for row in table], label="precision", marker="o")
    plt.plot(thresholds, [row["recall"] for row in table], label="recall", marker="o")
    plt.axhline(precision_floor, color="gray", linestyle="--", label="P_min")
    plt.axvline(chosen, color="red", linestyle=":", label="chosen")
    plt.title("Val precision/recall vs threshold")
    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(matrix, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center")
    plt.xticks([0, 1], ["normal", "defect"])
    plt.yticks([0, 1], ["normal", "defect"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def metric_lines(metrics):
    return [
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- PR-AUC: {metrics['pr_auc']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
    ]


def write_phase4_report(winner, table, chosen, precision_floor, test, baseline, predictions_path, path):
    lines = [
        "# Phase 4 Evaluation Report",
        "",
        f"**Locked model: Experiment {winner}.**",
        "",
        "## Threshold selection (VALIDATION only)",
        f"Rule: maximize defect recall subject to defect precision >= {precision_floor} "
        "(stated before inspecting the table).",
        "",
        "| threshold | precision | recall |",
        "|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} |")

    lines += [
        "",
        f"**Chosen threshold: {chosen['threshold']:.2f}** "
        f"(precision={chosen['precision']:.4f}, recall={chosen['recall']:.4f})",
        "",
        "## Test set (touched once, this evaluation)",
    ]
    lines += metric_lines(test)
    lines += [f"- Confusion matrix: {test['confusion_matrix']}", ""]
    lines += ["## Majority-class do-nothing baseline (test set)"]
    lines += metric_lines(baseline)
    lines += [
        "",
        "## Frozen predictions",
        f"Saved to: {predictions_path}",
        "Phase 5/6 only read this file, never change it.",
    ]
    path.write_text("\n".join(lines))
