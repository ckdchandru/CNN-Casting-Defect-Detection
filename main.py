import time

import pandas as pd
from torch.utils.data import DataLoader

from src.data import splitting, validation
from src.evaluation import evaluator
from src.explainability import failure_analysis, gradcam
from src.training import trainer
from src.data.dataset import CastingDataset, get_transforms
from src.models.resnet import load_model
from src.utils.helpers import (
    end_step,
    format_time,
    get_device,
    load_config,
    relative_path,
    save_to_config,
    set_seed,
    start_step,
)


def phase1(config):
    data = config["data"]
    figures = config["paths"]["figures"]

    t = start_step("1.1 --- Find Images")
    raw_df = validation.find_images(data["raw_dir"], data["class_folders"])
    print(f"Images found: {len(raw_df):,}")
    end_step(t)

    t = start_step("1.2 --- Raw Class Balance")
    raw = validation.class_summary(raw_df)
    print(f"Class counts: {raw['counts']}")
    print(f"Majority: {raw['majority']} ({raw['n_majority']:,})")
    print(f"Minority: {raw['minority']} ({raw['n_minority']:,})")
    print(f"Ratio: {raw['ratio']:.2f}:1")
    end_step(t)

    t = start_step("1.3 --- Exact Duplicates (MD5)")
    dup_groups = validation.find_exact_duplicates(raw_df)
    print(f"Duplicate groups: {len(dup_groups)}")
    end_step(t)

    t = start_step("1.4 --- Image Quality Check")
    quality = validation.check_images(raw_df)
    print(f"Unreadable images: {len(quality['broken'])}")
    print(f"Sizes seen: {quality['sizes']}")
    print(f"Modes seen: {quality['modes']}")
    if quality["broken"]:
        raise RuntimeError(f"{len(quality['broken'])} corrupt image(s) found")
    end_step(t)

    t = start_step("1.5 --- Build 4:1 Imbalanced Pool")
    # keep normals, subsample defect images
    pool_df = validation.make_imbalanced_pool(
        raw_df, "normal", "defect", data["target_ratio"], config["project"]["seed"]
    )
    print(f"Pool size: {len(pool_df):,}")
    end_step(t)

    t = start_step("1.6 --- Pool Class Balance")
    pool = validation.class_summary(pool_df)
    print(f"Class counts: {pool['counts']}")
    print(f"Ratio: {pool['ratio']:.2f}:1")
    print(f"Minority prevalence: {pool['prevalence']:.2%}")
    end_step(t)

    t = start_step("1.7 --- Gate Check")
    passed, message = validation.check_gate(pool)
    print(f"Gate passed: {passed} ({message})")
    if not passed:
        raise RuntimeError(f"Phase 1 gate FAILED: {message}")
    end_step(t)

    t = start_step("1.8 --- Save & Lock Dataset")
    validation.plot_class_counts(raw, "Class distribution - raw pool",
                                 figures / "phase1_class_distribution_raw.png")
    validation.plot_class_counts(pool, "Class distribution - constructed pool",
                                 figures / "phase1_class_distribution_constructed.png")
    validation.save_pool(pool_df, data["splits_dir"] / "constructed_pool.csv")
    validation.save_duplicate_groups(dup_groups, data["splits_dir"] / "duplicate_groups.csv")
    validation.write_phase1_report(raw, pool, len(dup_groups), passed, message,
                                   config["paths"]["reports"] / "phase1_gate_report.md")

    if not data["locked"]:
        save_to_config("data", "locked", True)
        save_to_config("data", "imbalance_ratio", round(pool["ratio"], 4))
        save_to_config("data", "minority_prevalence", round(pool["prevalence"], 4))
        data["locked"] = True
        print("Dataset LOCKED.")
    else:
        print("Dataset already locked.")
    end_step(t)

    return {"raw": raw, "pool": pool, "n_dup_groups": len(dup_groups)}


def phase2(config):
    data = config["data"]
    figures = config["paths"]["figures"]

    if not data["locked"]:
        raise RuntimeError("Dataset is not locked. Run Phase 1 first.")

    t = start_step("2.1 --- Load Pool")
    pool_df = pd.read_csv(data["splits_dir"] / "constructed_pool.csv")
    print(f"Pool loaded: {len(pool_df):,} images")
    end_step(t)

    t = start_step("2.2 --- Near-Duplicate Groups")
    grouped_df = splitting.add_group_ids(pool_df)
    sizes = grouped_df.groupby("group_id").size()
    print(f"Total groups: {len(sizes):,} (from {len(grouped_df):,} images)")
    print(f"Groups with >1 image: {(sizes > 1).sum():,}")
    print(f"Largest group: {sizes.max()} images")
    end_step(t)

    t = start_step("2.3 --- Group-Aware Stratified Split")
    split_df = splitting.split_by_group(
        grouped_df, data["train_frac"], data["val_frac"], data["test_frac"],
        config["project"]["seed"],
    )
    splitting.check_no_leakage(split_df)
    counts = splitting.split_counts(split_df)
    for (split_name, class_name), n in sorted(counts.items()):
        print(f"{split_name:<6} {class_name:<8}: {n:,}")
    print("No group leakage: confirmed")
    end_step(t)

    t = start_step("2.4 --- Save Split Files")
    split_paths = splitting.save_splits(split_df, data["splits_dir"])
    splitting.write_near_duplicate_report(
        grouped_df, config["paths"]["reports"] / "phase2_near_duplicate_report.md"
    )
    for path in split_paths.values():
        print(f"Saved: {relative_path(path)}")
    end_step(t)

    t = start_step("2.5 --- Transform Sanity Check")
    train_tf, _ = get_transforms(config["train"]["image_size"], config["augmentation"])
    train_data = CastingDataset(split_paths["train"], train_tf)
    image, label = train_data[0]
    print(f"Train size: {len(train_data):,}")
    print(f"Tensor shape: {tuple(image.shape)}, label: {label}")
    end_step(t)

    t = start_step("2.6 --- EDA Plots")
    splitting.plot_split_distribution(split_df, figures / "phase2_split_distribution.png")
    splitting.save_sample_grid(split_df[split_df["split"] == "train"], 5,
                               figures / "phase2_sample_grid.png")
    end_step(t)

    return {"counts": counts}


def phase3(config, device):
    figures = config["paths"]["figures"]
    seed = config["project"]["seed"]
    print(f"\nTraining device: {device}")

    results = {}
    for exp, title in [("A", "3.1 --- Experiment A (Plain CE)"),
                       ("B", "3.2 --- Experiment B (Weighted CE)")]:
        t = start_step(title)
        set_seed(seed)
        results[exp] = trainer.train_experiment(config, exp, device)
        best = results[exp]["best"]
        print(f"Best epoch: {best['epoch']} (stopped early: {results[exp]['stopped_early']})")
        print(f"Recall: {best['recall']:.4f} | PR-AUC: {best['pr_auc']:.4f} | F1: {best['f1']:.4f}")
        end_step(t)

    t = start_step("3.3 --- Pick Winner")
    winner, reason = trainer.pick_winner(results["A"], results["B"])
    save_to_config("train", "winner", winner)
    config["train"]["winner"] = winner
    print(f"Winner: Experiment {winner} ({reason})")
    end_step(t)

    t = start_step("3.4 --- Save Training Reports")
    for exp in ["A", "B"]:
        trainer.plot_training_curve(results[exp], exp, figures / f"phase3_experiment_{exp}_curve.png")
    trainer.write_phase3_report(results["A"], results["B"], winner, reason,
                              config["paths"]["reports"] / "phase3_model_selection_report.md")
    end_step(t)

    return {"A": results["A"], "B": results["B"], "winner": winner, "reason": reason}


def phase4(config, device):
    winner = config["train"]["winner"]
    paths = config["paths"]
    splits_dir = config["data"]["splits_dir"]
    batch_size = config["train"]["batch_size"]
    _, eval_tf = get_transforms(config["train"]["image_size"], config["augmentation"])

    t = start_step("4.1 --- Load Winning Model")
    model, checkpoint = load_model(config, winner, device)
    print(f"Model: Experiment {winner} ({relative_path(checkpoint)})")
    end_step(t)

    t = start_step("4.2 --- Threshold On Validation")
    val_loader = DataLoader(CastingDataset(splits_dir / "val.csv", eval_tf),
                            batch_size=batch_size, shuffle=False)
    val_labels, val_probs = evaluator.predict(model, val_loader, device)

    floor = config["evaluation"]["precision_floor"]
    table = evaluator.threshold_table(val_labels, val_probs, config["evaluation"]["threshold_candidates"])
    chosen = evaluator.choose_threshold(table, floor)
    threshold = chosen["threshold"]
    save_to_config("evaluation", "threshold", threshold)
    config["evaluation"]["threshold"] = threshold

    for row in table:
        print(f"threshold={row['threshold']:.2f}  precision={row['precision']:.4f}  recall={row['recall']:.4f}")
    print(f"Chosen threshold: {threshold:.2f}")
    end_step(t)

    t = start_step("4.3 --- Test Evaluation (Once)")
    test_csv = splits_dir / "test.csv"
    test_loader = DataLoader(CastingDataset(test_csv, eval_tf), batch_size=batch_size, shuffle=False)
    test_labels, test_probs = evaluator.predict(model, test_loader, device)
    test_preds = (test_probs >= threshold).astype(int)

    test = evaluator.get_metrics(test_labels, test_preds, test_probs)
    for name in ["precision", "recall", "f1", "pr_auc", "accuracy"]:
        print(f"{name}: {test[name]:.4f}")
    print(f"Confusion matrix: {test['confusion_matrix']}")
    end_step(t)

    t = start_step("4.4 --- Do-Nothing Baseline")
    baseline = evaluator.baseline_metrics(test_labels)
    print(f"Baseline recall: {baseline['recall']:.4f}, PR-AUC: {baseline['pr_auc']:.4f}, "
          f"accuracy: {baseline['accuracy']:.4f}")
    end_step(t)

    t = start_step("4.5 --- Save Predictions & Reports")
    predictions_path = paths["predictions"] / "test_predictions.csv"
    image_ids = pd.read_csv(test_csv)["filepath"].tolist()
    evaluator.save_predictions(image_ids, test_labels, test_preds, test_probs, threshold, predictions_path)
    evaluator.plot_threshold_curve(table, threshold, floor, paths["figures"] / "phase4_threshold_curve.png")
    evaluator.plot_confusion_matrix(test["confusion_matrix"], paths["figures"] / "phase4_confusion_matrix.png")
    evaluator.write_phase4_report(winner, table, chosen, floor, test, baseline,
                                 relative_path(predictions_path),
                                 paths["reports"] / "phase4_evaluation_report.md")
    end_step(t)

    return {"chosen": chosen, "test": test, "baseline": baseline}


def phase5(config, device):
    paths = config["paths"]
    image_size = config["train"]["image_size"]

    t = start_step("5.1 --- Grad-CAM Hooks")
    model, _ = load_model(config, config["train"]["winner"], device)
    cam = gradcam.GradCAM(model, config["model"]["gradcam_layer"], device)
    print(f"Target layer: {config['model']['gradcam_layer']}")
    end_step(t)

    t = start_step("5.2 --- Heatmaps For Every Test Image")
    predictions_df = pd.read_csv(paths["predictions"] / "test_predictions.csv")
    _, eval_tf = get_transforms(image_size, config["augmentation"])
    manifest_df = gradcam.make_all_heatmaps(cam, predictions_df, eval_tf, image_size, paths["gradcam"])
    manifest_df.to_csv(paths["gradcam"] / "gradcam_manifest.csv", index=False)
    print(f"Heatmaps generated: {len(manifest_df):,}")
    end_step(t)

    t = start_step("5.3 --- True-Positive Review Grid")
    grid_path = paths["figures"] / "phase5_true_positive_gate_sample.png"
    gradcam.true_positive_grid(manifest_df, 10, grid_path)
    print(f"Saved: {relative_path(grid_path)} (check visually)")
    end_step(t)

    t = start_step("5.4 --- Grad-CAM Report")
    review = config["gradcam_review"]
    gradcam.write_phase5_report(manifest_df, relative_path(grid_path), review["verdict"],
                                review["notes"], paths["reports"] / "phase5_gradcam_report.md")
    end_step(t)

    return {"n_heatmaps": len(manifest_df)}


def phase6(config):
    paths = config["paths"]

    t = start_step("6.1 --- Load Predictions")
    predictions_df = pd.read_csv(paths["predictions"] / "test_predictions.csv")
    manifest_df = pd.read_csv(paths["gradcam"] / "gradcam_manifest.csv")
    end_step(t)

    t = start_step("6.2 --- Collect Misclassifications")
    records_path = paths["reports"] / "phase6_failure_records.csv"
    failure_df = failure_analysis.find_failures(predictions_df, manifest_df)
    # don't erase annotations
    failure_df = failure_analysis.keep_old_annotations(failure_df, records_path)
    fp, fn = failure_analysis.count_fp_fn(failure_df)
    print(f"Misclassifications: {len(failure_df)} (FP={fp}, FN={fn})")
    end_step(t)

    t = start_step("6.3 --- Failure Grid & Report")
    grid_path = paths["figures"] / "phase6_failure_grid.png"
    failure_analysis.failure_grid(failure_df, grid_path)
    failure_df[failure_analysis.RECORD_COLUMNS].to_csv(records_path, index=False)
    failure_analysis.write_phase6_report(failure_df, relative_path(grid_path), relative_path(records_path),
                                 paths["reports"] / "phase6_failure_analysis_report.md")
    end_step(t)

    return {"n_failures": len(failure_df), "fp": fp, "fn": fn}


def print_summary(p1, p2, p3, p4, p5, p6, total_time):
    raw, pool = p1["raw"], p1["pool"]
    test, baseline, chosen = p4["test"], p4["baseline"], p4["chosen"]

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"Raw pool         : {raw['total']:,} images, {raw['ratio']:.2f}:1 ({raw['majority']} majority)")
    print(f"Constructed pool : {pool['total']:,} images, {pool['ratio']:.2f}:1 "
          f"({pool['prevalence']:.1%} {pool['minority']})")
    print(f"MD5 dup groups   : {p1['n_dup_groups']}")

    for name in ["train", "val", "test"]:
        n_defect = p2["counts"].get((name, "defect"), 0)
        n_total = n_defect + p2["counts"].get((name, "normal"), 0)
        print(f"{name:<6}           : {n_total:,} images ({n_defect:,} defect)")

    for exp in ["A", "B"]:
        best = p3[exp]["best"]
        print(f"Experiment {exp}     : recall={best['recall']:.4f} pr_auc={best['pr_auc']:.4f} f1={best['f1']:.4f}")
    print(f"Winner           : Experiment {p3['winner']} ({p3['reason']})")
    print(f"Threshold        : {chosen['threshold']:.2f}")
    print(f"TEST recall      : {test['recall']:.4f} (baseline {baseline['recall']:.4f})")
    print(f"TEST precision   : {test['precision']:.4f} (baseline {baseline['precision']:.4f})")
    print(f"TEST PR-AUC      : {test['pr_auc']:.4f} (baseline {baseline['pr_auc']:.4f})")
    print(f"Grad-CAM maps    : {p5['n_heatmaps']}")
    print(f"Misclassified    : {p6['n_failures']} (FP={p6['fp']}, FN={p6['fn']})")
    print(f"Total runtime    : {format_time(total_time)}")
    print("=" * 78)


def main():
    start = time.time()
    config = load_config()
    device = get_device()

    p1 = phase1(config)
    p2 = phase2(config)
    p3 = phase3(config, device)
    p4 = phase4(config, device)
    p5 = phase5(config, device)
    p6 = phase6(config)

    print_summary(p1, p2, p3, p4, p5, p6, time.time() - start)


if __name__ == "__main__":
    main()
