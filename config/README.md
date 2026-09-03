# config.yaml — key reference

| Key | Meaning |
|---|---|
| `project.seed` | Single seed fixed for torch/numpy/python everywhere, for reproducibility. |
| `data.raw_dir` | Folder the downloaded dataset is unzipped into. Update after Phase 1. |
| `data.class_folders` | Actual sub-folder names for each class. **Placeholders** — confirm against the real unzipped dataset in Phase 1, don't assume. |
| `data.locked` | Flipped to `true` only by `verify.py` once the Phase 1 imbalance gate passes. Everything downstream should check this before running. |
| `data.imbalance_ratio` / `minority_prevalence` | Written by Phase 1 verification, not hand-set. |
| `data.train_frac/val_frac/test_frac` | Stratified split proportions. Group-aware splitting overrides this if a grouping key is found. |
| `model.backbone` | `resnet18` is the primary, locked backbone. `efficientnet_b0` is optional/secondary — do not build both as co-primary. |
| `model.gradcam_target_layer` | Pinned Grad-CAM hook target, not searched for dynamically. For resnet18 this is the last BasicBlock's second conv in layer4. |
| `train.experiment` | Which experiment this training run is: `A` (naive baseline), `B` (weighted CE, primary), `C` (focal loss, optional). |
| `train.class_weight_method` | Method name only. The actual weight numbers are computed from TRAIN-split label counts at train time (see `src/model/losses.py`), never hand-typed here. |
| `evaluation.precision_floor` | P_min — the minimum acceptable defect-class precision. Must be decided **before** looking at the validation PR-curve table (Phase 4). |
| `evaluation.threshold_candidates` | Candidate thresholds swept to build the PR-curve table. |
| `paths.*` | Where each phase's outputs get written — checkpoints, figures, Grad-CAM overlays, saved test predictions, reports. |
