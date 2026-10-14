import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.dataset import DEFECT, CastingDataset, get_transforms
from src.models.resnet import build_model, get_class_weights, get_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_labels += labels.cpu().tolist()
            all_preds += probs.argmax(dim=1).cpu().tolist()
            all_probs += probs[:, DEFECT].cpu().tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[DEFECT], average=None, zero_division=0
    )
    pr_auc = average_precision_score(all_labels, all_probs)

    return {
        "loss": total_loss / len(loader.dataset),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "pr_auc": float(pr_auc),
    }


def train_experiment(config, experiment, device):
    splits_dir = config["data"]["splits_dir"]
    train_csv = splits_dir / "train.csv"
    settings = config["train"]

    train_tf, eval_tf = get_transforms(settings["image_size"], config["augmentation"])
    train_data = CastingDataset(train_csv, train_tf)
    val_data = CastingDataset(splits_dir / "val.csv", eval_tf)

    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=settings["batch_size"], shuffle=False)

    model = build_model(config["model"]["num_classes"], config["model"]["pretrained"]).to(device)

    class_weights = None
    if experiment == "B":
        class_weights = get_class_weights(train_csv).to(device)
        print(f"Class weights: {class_weights.tolist()}")

    criterion = get_loss(experiment, class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    checkpoint = config["paths"]["checkpoints"] / f"experiment_{experiment}.pt"

    history = []
    best = {"epoch": 0, "recall": 0.0, "pr_auc": 0.0, "f1": 0.0, "precision": 0.0}
    bad_epochs = 0
    stopped_early = False

    for epoch in range(1, settings["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = validate(model, val_loader, criterion, device)
        val["epoch"] = epoch
        val["train_loss"] = train_loss
        history.append(val)

        print(f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
              f"| recall {val['recall']:.4f} | pr_auc {val['pr_auc']:.4f}")

        # checkpoint
        if val["pr_auc"] > best["pr_auc"]:
            best = {
                "epoch": epoch,
                "recall": val["recall"],
                "pr_auc": val["pr_auc"],
                "f1": val["f1"],
                "precision": val["precision"],
            }
            torch.save(model.state_dict(), checkpoint)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= settings["patience"]:
            stopped_early = True
            break

    return {"history": history, "best": best, "stopped_early": stopped_early}


def pick_winner(result_a, result_b):
    # recall, then PR-AUC, then F1
    for metric in ["recall", "pr_auc", "f1"]:
        a = result_a["best"][metric]
        b = result_b["best"][metric]
        if a != b:
            winner = "A" if a > b else "B"
            return winner, f"{metric}: A={a:.4f} vs B={b:.4f}"
    return "A", "exact tie on all three metrics"


def plot_training_curve(result, experiment, path):
    epochs = [h["epoch"] for h in result["history"]]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["recall"] for h in result["history"]], label="val_recall", marker="o")
    plt.plot(epochs, [h["pr_auc"] for h in result["history"]], label="val_pr_auc", marker="o")
    plt.plot(epochs, [h["f1"] for h in result["history"]], label="val_f1", marker="o")
    plt.axvline(result["best"]["epoch"], color="gray", linestyle="--", label="best epoch")
    plt.title(f"Experiment {experiment} - validation metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def experiment_lines(title, result):
    best = result["best"]
    return [
        title,
        f"- Best epoch: {best['epoch']} (stopped early: {result['stopped_early']})",
        f"- Recall: {best['recall']:.4f}",
        f"- PR-AUC: {best['pr_auc']:.4f}",
        f"- F1: {best['f1']:.4f}",
        f"- Precision: {best['precision']:.4f}",
    ]


def write_phase3_report(result_a, result_b, winner, reason, path):
    lines = ["# Phase 3 Model Selection Report", "", f"**Winner: Experiment {winner}.** {reason}", ""]
    lines += experiment_lines("## Experiment A (naive, unweighted CE)", result_a)
    lines += [""]
    lines += experiment_lines("## Experiment B (weighted CE, PRIMARY)", result_b)
    lines += [
        "",
        "## Selection rule",
        "PRIMARY = defect recall, SECONDARY = defect PR-AUC, TERTIARY = defect F1.",
        "Both experiments reported in full regardless of outcome.",
    ]
    path.write_text("\n".join(lines))
