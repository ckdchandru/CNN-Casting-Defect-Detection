import time
from pathlib import Path

from src.config import load_config
from src.data.discovery import DatasetDiscovery
from src.data.imbalance_analyzer import ImbalanceAnalyzer
from src.data.duplicate_checker import DuplicateChecker
from src.data.image_quality_checker import ImageQualityChecker
from src.data.imbalance_constructor import ImbalanceConstructor
from src.data.gate_validator import Phase1GateValidator
from src.data.dataset_locker import DatasetLocker
from src.data.report_writer import Phase1ReportWriter
from src.data.near_duplicate_detector import NearDuplicateDetector
from src.data.split import StratifiedGroupSplitter, assert_no_group_leakage
from src.data.split_writer import Phase2SplitWriter
from src.data.transforms import build_transforms
from src.data.dataset import CastingDataset
from src.data.eda import EDAReporter
from src.train.run_experiment import ExperimentRunner
from src.train.model_selector import ModelSelector
from src.train.report_writer import Phase3ReportWriter
from src.model.backbone import build_backbone
from src.evaluate.predictor import Predictor
from src.evaluate.threshold_selector import ThresholdSelector
from src.evaluate.test_evaluator import TestEvaluator
from src.evaluate.baseline import MajorityBaseline
from src.evaluate.predictions_writer import PredictionsWriter
from src.evaluate.report_writer import Phase4ReportWriter
from src.explain.gradcam import GradCAM
from src.explain.heatmap_batch_runner import HeatmapBatchRunner
from src.explain.report_writer import Phase5ReportWriter
from src.explain.failure_analyzer import FailureAnalyzer
from src.explain.failure_report_writer import Phase6ReportWriter
from src.utils.formatting import print_stage_header, print_elapsed
from src.utils.seed import set_seed
from src.utils.config_updater import update_config_field

import pandas as pd
import torch

STAGE_TITLES = [
    "1.1 --- Dataset Discovery",
    "1.2 --- Raw Imbalance Analysis",
    "1.3 --- Duplicate Detection",
    "1.4 --- Image Quality Check",
    "1.5 --- Imbalance Construction",
    "1.6 --- Constructed Pool Analysis",
    "1.7 --- Gate Validation",
    "1.8 --- Lock & Save Reports",
    "2.1 --- Load Constructed Pool",
    "2.2 --- Near-Duplicate Detection",
    "2.3 --- Stratified Group Split",
    "2.4 --- Write Split Files",
    "2.5 --- Build & Sanity-Check Transforms",
    "2.6 --- EDA: Distribution & Samples",
    "3.1 --- Train Experiment A (Naive)",
    "3.2 --- Train Experiment B (Weighted CE)",
    "3.3 --- Model Selection (A vs B)",
    "3.4 --- Save Training Reports",
    "4.1 --- Load Locked Model",
    "4.2 --- Threshold Selection (VAL)",
    "4.3 --- Test Evaluation (ONE-TIME)",
    "4.4 --- Do-Nothing Baseline",
    "4.5 --- Save Predictions & Reports",
    "5.1 --- Build Grad-CAM Hooks",
    "5.2 --- Generate Heatmaps (Every Test Row)",
    "5.3 --- True-Positive Gate Sample",
    "5.4 --- Save Grad-CAM Report",
    "6.1 --- Load Misclassifications (Post-Hoc)",
    "6.2 --- Build Failure Records",
    "6.3 --- Save Failure Grid & Report",
]


def main():
    pipeline_start = time.time()
    config = load_config("config/config.yaml")

    phase1_results = run_phase1(config)
    phase2_results = run_phase2(config)
    phase3_results = run_phase3(config)
    phase4_results = run_phase4(config, phase3_results)
    phase5_results = run_phase5(config, phase3_results)
    phase6_results = run_phase6(config)

    print_summary(phase1_results, phase2_results, phase3_results, phase4_results, time.time() - pipeline_start)
    print_phase5_summary(phase5_results)
    print_phase6_summary(phase6_results)


def run_phase1(config: dict) -> dict:
    """Stages 1.1-1.8: verify, construct, lock."""

    # 1.1 - discovery
    print_stage_header(STAGE_TITLES[0], 1, len(STAGE_TITLES))
    stage_start = time.time()

    discovery = DatasetDiscovery(config["data"]["raw_dir"], config["data"]["class_folders"])
    df_raw = discovery.discover()

    print(f"Images found: {len(df_raw):,}")
    print_elapsed(stage_start)

    # 1.2 - raw imbalance
    print_stage_header(STAGE_TITLES[1], 2, len(STAGE_TITLES))
    stage_start = time.time()

    analyzer = ImbalanceAnalyzer()
    raw_report = analyzer.analyze(df_raw)

    print(f"Class counts: {raw_report.class_counts}")
    print(f"Majority: {raw_report.majority_class} ({raw_report.n_majority:,})")
    print(f"Minority: {raw_report.minority_class} ({raw_report.n_minority:,})")
    print(f"Ratio: {raw_report.imbalance_ratio:.2f}:1")
    print_elapsed(stage_start)

    # 1.3 - duplicate check
    print_stage_header(STAGE_TITLES[2], 3, len(STAGE_TITLES))
    stage_start = time.time()

    duplicate_checker = DuplicateChecker()
    duplicates = duplicate_checker.check(df_raw)

    print(f"Duplicate groups: {duplicates.n_duplicate_groups}")
    print_elapsed(stage_start)

    # 1.4 - image quality
    print_stage_header(STAGE_TITLES[3], 4, len(STAGE_TITLES))
    stage_start = time.time()

    quality_checker = ImageQualityChecker()
    quality = quality_checker.check(df_raw)

    print(f"Unreadable images: {quality.n_unreadable}")
    print(f"Sizes seen: {quality.sizes_seen}")
    print(f"Modes seen: {quality.modes_seen}")

    if quality.n_unreadable:
        raise RuntimeError(f"{quality.n_unreadable} corrupt image(s) found")

    print_elapsed(stage_start)

    # 1.5 - construct imbalance
    print_stage_header(STAGE_TITLES[4], 5, len(STAGE_TITLES))
    stage_start = time.time()

    constructor = ImbalanceConstructor(
        target_ratio=config["data"]["target_majority_minority_ratio"],
        seed=config["project"]["seed"],
    )
    df_constructed = constructor.construct(
        df_raw, majority_class="normal", minority_class="defect"
    )

    print(f"Constructed pool size: {len(df_constructed):,}")
    print_elapsed(stage_start)

    # 1.6 - constructed imbalance
    print_stage_header(STAGE_TITLES[5], 6, len(STAGE_TITLES))
    stage_start = time.time()

    constructed_report = analyzer.analyze(df_constructed)

    print(f"Class counts: {constructed_report.class_counts}")
    print(f"Ratio: {constructed_report.imbalance_ratio:.2f}:1")
    print(f"Minority prevalence: {constructed_report.minority_prevalence:.2%}")
    print_elapsed(stage_start)

    # 1.7 - gate validation
    print_stage_header(STAGE_TITLES[6], 7, len(STAGE_TITLES))
    stage_start = time.time()

    gate_validator = Phase1GateValidator()
    gate_result = gate_validator.validate(constructed_report)

    print(f"Gate passed: {gate_result.is_valid}")
    print(f"Reason: {gate_result.message}")

    if not gate_result.is_valid:
        raise RuntimeError(f"Phase 1 gate FAILED: {gate_result.message}")

    print_elapsed(stage_start)

    # 1.8 - lock and save
    print_stage_header(STAGE_TITLES[7], 8, len(STAGE_TITLES))
    stage_start = time.time()

    writer = Phase1ReportWriter(
        figures_dir=config["paths"]["figures_dir"],
        splits_dir=config["data"]["splits_dir"],
        reports_dir=config["paths"]["reports_dir"],
    )
    writer.save_distribution_plot(
        raw_report, "Class distribution - raw pool", "phase1_class_distribution_raw.png"
    )
    writer.save_distribution_plot(
        constructed_report,
        "Class distribution - constructed pool",
        "phase1_class_distribution_constructed.png",
    )
    pool_path = writer.save_constructed_pool(df_constructed)
    dupes_path = writer.save_duplicate_groups(duplicates)
    report_path = writer.write_gate_report(
        raw_report, constructed_report, duplicates, gate_result
    )

    if not config["data"].get("locked", False):
        locker = DatasetLocker()
        locker.lock(config_path="config/config.yaml", report=constructed_report)
        # Sync in-memory config too.
        # File write alone isn't enough.
        config["data"]["locked"] = True
        config["data"]["imbalance_ratio"] = round(constructed_report.imbalance_ratio, 4)
        config["data"]["minority_prevalence"] = round(
            constructed_report.minority_prevalence, 4
        )
        print("Dataset LOCKED.")
    else:
        print("Dataset already locked, config.yaml unchanged.")

    print(f"Saved: {pool_path}")
    print(f"Saved: {dupes_path}")
    print(f"Saved: {report_path}")
    print_elapsed(stage_start)

    return {
        "raw_report": raw_report,
        "constructed_report": constructed_report,
        "duplicates": duplicates,
        "pool_path": pool_path,
        "dupes_path": dupes_path,
    }


def run_phase2(config: dict) -> dict:
    """Stages 2.1-2.6: near-dup groups, split, transforms, EDA."""

    if not config["data"].get("locked", False):
        raise RuntimeError("Phase 2 requires a LOCKED dataset. Run Phase 1 first.")

    # 2.1 - load constructed pool
    print_stage_header(STAGE_TITLES[8], 9, len(STAGE_TITLES))
    stage_start = time.time()

    pool_df = pd.read_csv(config["data"]["splits_dir"] / "constructed_pool.csv")

    print(f"Constructed pool loaded: {len(pool_df):,} images")
    print_elapsed(stage_start)

    # 2.2 - near-duplicate detection
    print_stage_header(STAGE_TITLES[9], 10, len(STAGE_TITLES))
    stage_start = time.time()

    detector = NearDuplicateDetector()
    grouped_df = detector.detect(pool_df)

    group_sizes = grouped_df.groupby("group_id").size()
    print(f"Total groups: {grouped_df['group_id'].nunique():,} (from {len(grouped_df):,} images)")
    print(f"Groups with >1 image: {(group_sizes > 1).sum():,}")
    print(f"Largest group: {group_sizes.max()} images")
    print_elapsed(stage_start)

    # 2.3 - stratified group split
    print_stage_header(STAGE_TITLES[10], 11, len(STAGE_TITLES))
    stage_start = time.time()

    splitter = StratifiedGroupSplitter(
        train_frac=config["data"]["train_frac"],
        val_frac=config["data"]["val_frac"],
        test_frac=config["data"]["test_frac"],
        seed=config["project"]["seed"],
    )
    merged_df, diagnostics = splitter.split(grouped_df)
    assert_no_group_leakage(merged_df)

    for (split_name, class_name), count in sorted(diagnostics.items()):
        print(f"{split_name:<6} {class_name:<8}: {count:,}")
    print("No group leakage across splits: confirmed")
    print_elapsed(stage_start)

    # 2.4 - write split files
    print_stage_header(STAGE_TITLES[11], 12, len(STAGE_TITLES))
    stage_start = time.time()

    split_writer = Phase2SplitWriter(config["data"]["splits_dir"])
    split_paths = split_writer.write(merged_df)
    near_dup_report_path = split_writer.write_near_duplicate_report(
        grouped_df, config["paths"]["reports_dir"]
    )

    for split_name, path in split_paths.items():
        print(f"Saved: {path}")
    print(f"Saved: {near_dup_report_path}")
    print_elapsed(stage_start)

    # 2.5 - transforms, sanity check
    print_stage_header(STAGE_TITLES[12], 13, len(STAGE_TITLES))
    stage_start = time.time()

    transform_pipelines = build_transforms(
        image_size=config["train"]["image_size"],
        augmentation_config=config["augmentation"],
    )
    train_dataset = CastingDataset(split_paths["train"], transform=transform_pipelines["train"])
    eval_dataset = CastingDataset(split_paths["val"], transform=transform_pipelines["eval"])

    sample_image, sample_label = train_dataset[0]
    print(f"Train dataset size: {len(train_dataset):,}")
    print(f"Sample tensor shape: {tuple(sample_image.shape)}")
    print(f"Sample label: {sample_label}")
    print_elapsed(stage_start)

    # 2.6 - EDA
    print_stage_header(STAGE_TITLES[13], 14, len(STAGE_TITLES))
    stage_start = time.time()

    eda = EDAReporter(figures_dir=config["paths"]["figures_dir"])
    dist_path = eda.plot_split_distribution(merged_df, "phase2_split_distribution.png")
    sample_path = eda.save_sample_grid(
        merged_df[merged_df["split"] == "train"], n_per_class=5, filename="phase2_sample_grid.png"
    )

    print(f"Saved: {dist_path}")
    print(f"Saved: {sample_path}")
    print_elapsed(stage_start)

    return {
        "merged_df": merged_df,
        "diagnostics": diagnostics,
        "split_paths": split_paths,
    }


def run_phase3(config: dict) -> dict:
    """Stages 3.1-3.4: Experiment A vs B, locked tie-break."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}")

    # 3.1 - Experiment A
    print_stage_header(STAGE_TITLES[14], 15, len(STAGE_TITLES))
    stage_start = time.time()

    set_seed(config["project"]["seed"])
    runner = ExperimentRunner(config, device)
    result_a = runner.run("A")

    print(f"Best epoch: {result_a.best_epoch} (stopped early: {result_a.stopped_early})")
    print(f"Recall: {result_a.best_val_recall:.4f}")
    print(f"PR-AUC: {result_a.best_val_pr_auc:.4f}")
    print(f"F1: {result_a.best_val_f1:.4f}")
    print_elapsed(stage_start)

    # 3.2 - Experiment B
    print_stage_header(STAGE_TITLES[15], 16, len(STAGE_TITLES))
    stage_start = time.time()

    set_seed(config["project"]["seed"])
    result_b = runner.run("B")

    print(f"Best epoch: {result_b.best_epoch} (stopped early: {result_b.stopped_early})")
    print(f"Recall: {result_b.best_val_recall:.4f}")
    print(f"PR-AUC: {result_b.best_val_pr_auc:.4f}")
    print(f"F1: {result_b.best_val_f1:.4f}")
    print_elapsed(stage_start)

    # 3.3 - model selection
    print_stage_header(STAGE_TITLES[16], 17, len(STAGE_TITLES))
    stage_start = time.time()

    selector = ModelSelector()
    selection = selector.select(result_a, result_b)

    update_config_field(Path("config/config.yaml"), "train", "locked_winner", selection.winner)
    config["train"]["locked_winner"] = selection.winner

    print(f"Winner: Experiment {selection.winner}")
    print(f"Reason: {selection.reason}")
    print_elapsed(stage_start)

    # 3.4 - save reports
    print_stage_header(STAGE_TITLES[17], 18, len(STAGE_TITLES))
    stage_start = time.time()

    writer = Phase3ReportWriter(
        figures_dir=config["paths"]["figures_dir"],
        reports_dir=config["paths"]["reports_dir"],
    )
    curve_a_path = writer.save_training_curve(result_a, "A")
    curve_b_path = writer.save_training_curve(result_b, "B")
    report_path = writer.write_comparison_report(result_a, result_b, selection)

    print(f"Saved: {curve_a_path}")
    print(f"Saved: {curve_b_path}")
    print(f"Saved: {report_path}")
    print_elapsed(stage_start)

    return {
        "result_a": result_a,
        "result_b": result_b,
        "selection": selection,
    }


def load_locked_model(config: dict, winner: str, device: torch.device) -> torch.nn.Module:
    """Rebuild architecture, load the winning checkpoint."""
    model = build_backbone(
        name=config["model"]["backbone"],
        pretrained=False,
        num_classes=config["model"]["num_classes"],
    ).to(device)
    checkpoint_path = config["paths"]["checkpoints_dir"] / f"experiment_{winner}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, checkpoint_path


def run_phase4(config: dict, phase3_results: dict) -> dict:
    """Stages 4.1-4.5: threshold on VAL, one-time TEST eval."""

    winner = phase3_results["selection"].winner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1 - load locked model
    print_stage_header(STAGE_TITLES[18], 19, len(STAGE_TITLES))
    stage_start = time.time()

    model, checkpoint_path = load_locked_model(config, winner, device)

    print(f"Locked model: Experiment {winner}")
    print(f"Checkpoint: {checkpoint_path}")
    print_elapsed(stage_start)

    # 4.2 - threshold on VAL
    print_stage_header(STAGE_TITLES[19], 20, len(STAGE_TITLES))
    stage_start = time.time()

    transform_pipelines = build_transforms(
        image_size=config["train"]["image_size"],
        augmentation_config=config["augmentation"],
    )
    val_dataset = CastingDataset(
        config["data"]["splits_dir"] / "val.csv", transform=transform_pipelines["eval"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["train"]["batch_size"], shuffle=False
    )

    predictor = Predictor(model, device)
    val_labels, val_probs = predictor.predict(val_loader)

    selector = ThresholdSelector(
        precision_floor=config["evaluation"]["precision_floor"],
        candidates=config["evaluation"]["threshold_candidates"],
    )
    threshold_result = selector.select(val_labels, val_probs)

    update_config_field(
        Path("config/config.yaml"), "evaluation", "locked_threshold", threshold_result.chosen_threshold
    )
    config["evaluation"]["locked_threshold"] = threshold_result.chosen_threshold

    for row in threshold_result.table:
        print(f"threshold={row.threshold:.2f}  precision={row.precision:.4f}  recall={row.recall:.4f}")
    print(f"Chosen threshold: {threshold_result.chosen_threshold:.2f} "
          f"(precision={threshold_result.chosen_precision:.4f}, "
          f"recall={threshold_result.chosen_recall:.4f})")
    print_elapsed(stage_start)

    # 4.3 - ONE-TIME test evaluation
    print_stage_header(STAGE_TITLES[20], 21, len(STAGE_TITLES))
    stage_start = time.time()

    test_csv = config["data"]["splits_dir"] / "test.csv"
    test_dataset = CastingDataset(test_csv, transform=transform_pipelines["eval"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["train"]["batch_size"], shuffle=False
    )

    test_labels, test_probs = predictor.predict(test_loader)
    test_preds = (test_probs >= threshold_result.chosen_threshold).astype(int)

    evaluator = TestEvaluator()
    test_metrics = evaluator.evaluate(test_labels, test_preds, test_probs)

    print(f"Precision: {test_metrics.precision:.4f}")
    print(f"Recall: {test_metrics.recall:.4f}")
    print(f"F1: {test_metrics.f1:.4f}")
    print(f"PR-AUC: {test_metrics.pr_auc:.4f}")
    print(f"Accuracy (secondary): {test_metrics.accuracy:.4f}")
    print(f"Confusion matrix: {test_metrics.confusion_matrix}")
    print_elapsed(stage_start)

    # 4.4 - do-nothing baseline
    print_stage_header(STAGE_TITLES[21], 22, len(STAGE_TITLES))
    stage_start = time.time()

    baseline_metrics = MajorityBaseline().evaluate(test_labels)

    print(f"Baseline recall: {baseline_metrics.recall:.4f}")
    print(f"Baseline precision: {baseline_metrics.precision:.4f}")
    print(f"Baseline F1: {baseline_metrics.f1:.4f}")
    print(f"Baseline accuracy: {baseline_metrics.accuracy:.4f}")
    print_elapsed(stage_start)

    # 4.5 - save outputs
    print_stage_header(STAGE_TITLES[22], 23, len(STAGE_TITLES))
    stage_start = time.time()

    test_df = pd.read_csv(test_csv)
    predictions_writer = PredictionsWriter()
    predictions_path = predictions_writer.write(
        filepaths=test_df["filepath"].tolist(),
        true_labels=test_labels,
        preds=test_preds,
        probs=test_probs,
        threshold=threshold_result.chosen_threshold,
        predictions_dir=config["paths"]["predictions_dir"],
    )

    report_writer = Phase4ReportWriter(
        figures_dir=config["paths"]["figures_dir"],
        reports_dir=config["paths"]["reports_dir"],
    )
    pr_curve_path = report_writer.save_pr_curve_plot(threshold_result)
    confusion_path = report_writer.save_confusion_matrix_plot(
        test_metrics, "phase4_confusion_matrix.png"
    )
    report_path = report_writer.write_report(
        winner, threshold_result, test_metrics, baseline_metrics, predictions_path
    )

    print(f"Saved: {predictions_path}")
    print(f"Saved: {pr_curve_path}")
    print(f"Saved: {confusion_path}")
    print(f"Saved: {report_path}")
    print_elapsed(stage_start)

    return {
        "threshold_result": threshold_result,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "predictions_path": predictions_path,
    }


def run_phase5(config: dict, phase3_results: dict) -> dict:
    """Stages 5.1-5.4: Grad-CAM, post-hoc, every test row."""

    winner = phase3_results["selection"].winner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5.1 - build Grad-CAM hooks
    print_stage_header(STAGE_TITLES[23], 24, len(STAGE_TITLES))
    stage_start = time.time()

    model, checkpoint_path = load_locked_model(config, winner, device)
    gradcam = GradCAM(model, config["model"]["gradcam_target_layer"], device)

    print(f"Locked model (post-hoc, read-only): Experiment {winner}")
    print(f"Target layer: {config['model']['gradcam_target_layer']}")
    print_elapsed(stage_start)

    # 5.2 - generate heatmaps
    print_stage_header(STAGE_TITLES[24], 25, len(STAGE_TITLES))
    stage_start = time.time()

    predictions_path = config["paths"]["predictions_dir"] / "test_predictions.csv"
    predictions_df = pd.read_csv(predictions_path)

    transform_pipelines = build_transforms(
        image_size=config["train"]["image_size"],
        augmentation_config=config["augmentation"],
    )
    runner = HeatmapBatchRunner(
        gradcam=gradcam,
        eval_transform=transform_pipelines["eval"],
        image_size=config["train"]["image_size"],
        gradcam_dir=config["paths"]["gradcam_dir"],
    )
    manifest_df = runner.run(predictions_df)

    print(f"Heatmaps generated: {len(manifest_df):,} (read {predictions_path}, never rewritten)")
    print_elapsed(stage_start)

    # 5.3 - true-positive gate sample
    print_stage_header(STAGE_TITLES[25], 26, len(STAGE_TITLES))
    stage_start = time.time()

    report_writer = Phase5ReportWriter(
        figures_dir=config["paths"]["figures_dir"],
        reports_dir=config["paths"]["reports_dir"],
    )
    manifest_path = report_writer.save_manifest(manifest_df, config["paths"]["gradcam_dir"])
    grid_path = report_writer.save_true_positive_grid(
        manifest_df, n_samples=10, filename="phase5_true_positive_gate_sample.png"
    )

    print(f"Saved: {manifest_path}")
    print(f"Saved: {grid_path}")
    print("GATE: review this grid visually before treating Grad-CAM as trustworthy.")
    print_elapsed(stage_start)

    # 5.4 - save Grad-CAM report
    print_stage_header(STAGE_TITLES[26], 27, len(STAGE_TITLES))
    stage_start = time.time()

    report_path = report_writer.write_report(
        manifest_df,
        grid_path,
        gate_verdict=config["explainability"]["gate_5_3_verdict"],
        gate_notes=config["explainability"]["gate_5_3_notes"],
    )

    manifest_path = config["paths"]["gradcam_dir"] / "gradcam_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Saved: {report_path}")
    print(f"Saved: {manifest_path}")
    print_elapsed(stage_start)

    return {
        "manifest_df": manifest_df,
        "grid_path": grid_path,
        "report_path": report_path,
    }


def print_phase5_summary(phase5_results: dict) -> None:
    """Grad-CAM stage summary, pending human review."""
    manifest_df = phase5_results["manifest_df"]
    print()
    print("=" * 78)
    print("PHASE 5 SUMMARY")
    print("=" * 78)
    print(f"Heatmaps generated : {len(manifest_df):,}")
    print(f"Gate sample grid   : {phase5_results['grid_path']}")
    print(f"Report             : {phase5_results['report_path']}")
    print("\nPHASE 5 COMPLETE (mechanically). Gate 5.3 needs human visual review")
    print("of the sample grid before Phase 6 (failure analysis) begins.")
    print("=" * 78)


def run_phase6(config: dict) -> dict:
    """Stages 6.1-6.3: every misclassification, post-hoc only."""

    # 6.1 - load misclassifications (post-hoc)
    print_stage_header(STAGE_TITLES[27], 28, len(STAGE_TITLES))
    stage_start = time.time()

    predictions_path = config["paths"]["predictions_dir"] / "test_predictions.csv"
    manifest_path = config["paths"]["gradcam_dir"] / "gradcam_manifest.csv"
    predictions_df = pd.read_csv(predictions_path)
    manifest_df = pd.read_csv(manifest_path)

    print(f"Read: {predictions_path}")
    print(f"Read: {manifest_path}")
    print_elapsed(stage_start)

    # 6.2 - build failure records
    print_stage_header(STAGE_TITLES[28], 29, len(STAGE_TITLES))
    stage_start = time.time()

    analyzer = FailureAnalyzer()
    fresh_df = analyzer.analyze(predictions_df, manifest_df)

    # Preserve human-filled categories.
    # Rerun must not erase them.
    records_path = config["paths"]["reports_dir"] / "phase6_failure_records.csv"
    failure_df = analyzer.upsert_with_existing(fresh_df, records_path)

    n_fp = len(failure_df[(failure_df["true_label"] == 0) & (failure_df["predicted_label"] == 1)])
    n_fn = len(failure_df[(failure_df["true_label"] == 1) & (failure_df["predicted_label"] == 0)])
    n_annotated = len(failure_df[failure_df["failure_category"].fillna("") != ""])
    print(f"Total misclassifications: {len(failure_df)}")
    print(f"False positives: {n_fp}")
    print(f"False negatives: {n_fn}")
    print(f"Already annotated (preserved from disk): {n_annotated}")
    print_elapsed(stage_start)

    # 6.3 - save failure outputs
    print_stage_header(STAGE_TITLES[29], 30, len(STAGE_TITLES))
    stage_start = time.time()

    writer = Phase6ReportWriter(
        figures_dir=config["paths"]["figures_dir"],
        reports_dir=config["paths"]["reports_dir"],
    )
    grid_path = writer.save_failure_grid(failure_df, "phase6_failure_grid.png")
    records_path = writer.save_failure_records(failure_df)
    report_path = writer.write_report(failure_df, grid_path, records_path)

    print(f"Saved: {grid_path}")
    print(f"Saved: {records_path}")
    print(f"Saved: {report_path}")
    print_elapsed(stage_start)

    return {
        "failure_df": failure_df,
        "grid_path": grid_path,
        "records_path": records_path,
        "report_path": report_path,
    }


def print_phase6_summary(phase6_results: dict) -> None:
    """Failure analysis summary, pending human review."""
    failure_df = phase6_results["failure_df"]
    print()
    print("=" * 78)
    print("PHASE 6 SUMMARY")
    print("=" * 78)
    print(f"Total misclassifications : {len(failure_df)}")
    print(f"Failure grid             : {phase6_results['grid_path']}")
    print(f"Records CSV              : {phase6_results['records_path']}")
    print(f"Report                   : {phase6_results['report_path']}")
    print("\nPHASE 6 COMPLETE (mechanically). Categorization needs human visual")
    print("review of the failure grid before this project's analysis is final.")
    print("=" * 78)


def print_summary(phase1_results: dict, phase2_results: dict, phase3_results: dict, phase4_results: dict, total_elapsed: float) -> None:
    """Final run summary, all four phases."""
    raw_report = phase1_results["raw_report"]
    constructed_report = phase1_results["constructed_report"]
    duplicates = phase1_results["duplicates"]
    diagnostics = phase2_results["diagnostics"]
    result_a = phase3_results["result_a"]
    result_b = phase3_results["result_b"]
    selection = phase3_results["selection"]
    threshold_result = phase4_results["threshold_result"]
    test_metrics = phase4_results["test_metrics"]
    baseline_metrics = phase4_results["baseline_metrics"]

    print()
    print("=" * 78)
    print("PHASE 1 + 2 + 3 + 4 SUMMARY")
    print("=" * 78)
    print(f"Raw pool          : {raw_report.n_total:,} images, "
          f"{raw_report.imbalance_ratio:.2f}:1 ({raw_report.majority_class} majority)")
    print(f"Constructed pool  : {constructed_report.n_total:,} images, "
          f"{constructed_report.imbalance_ratio:.2f}:1 "
          f"({constructed_report.minority_prevalence:.1%} {constructed_report.minority_class})")
    print(f"Duplicate groups  : {duplicates.n_duplicate_groups} (exact-MD5, Phase 1)")

    for split_name in ("train", "val", "test"):
        n_normal = diagnostics.get((split_name, "normal"), 0)
        n_defect = diagnostics.get((split_name, "defect"), 0)
        total = n_normal + n_defect
        prevalence = n_defect / total if total else 0
        print(f"{split_name:<6}            : {total:,} images "
              f"({n_defect:,} defect, {prevalence:.1%})")

    print(f"Experiment A      : recall={result_a.best_val_recall:.4f} "
          f"pr_auc={result_a.best_val_pr_auc:.4f} f1={result_a.best_val_f1:.4f}")
    print(f"Experiment B      : recall={result_b.best_val_recall:.4f} "
          f"pr_auc={result_b.best_val_pr_auc:.4f} f1={result_b.best_val_f1:.4f}")
    print(f"Locked winner     : Experiment {selection.winner} ({selection.reason})")
    print(f"Chosen threshold  : {threshold_result.chosen_threshold:.2f} "
          f"(val precision={threshold_result.chosen_precision:.4f}, "
          f"recall={threshold_result.chosen_recall:.4f})")
    print(f"TEST recall       : {test_metrics.recall:.4f} "
          f"(baseline: {baseline_metrics.recall:.4f})")
    print(f"TEST precision    : {test_metrics.precision:.4f} "
          f"(baseline: {baseline_metrics.precision:.4f})")
    print(f"TEST PR-AUC       : {test_metrics.pr_auc:.4f} "
          f"(baseline: {baseline_metrics.pr_auc:.4f})")

    print(f"Total runtime     : {total_elapsed:.2f}s")
    print("\nPHASE 1 + 2 + 3 + 4 COMPLETE. Ready for Phase 5.")
    print("=" * 78)


if __name__ == "__main__":
    main()
