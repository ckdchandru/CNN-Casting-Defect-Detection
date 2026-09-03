# Phase 5 Grad-CAM Report

- Heatmaps generated: 568 (every test prediction)
- Correct predictions: 565
- True positives available for gate review: 123
- Sample grid for gate review: D:\Placements\ML_Projects\Cnn\outputs\figures\phase5_true_positive_gate_sample.png

## What Grad-CAM does and does not prove
Grad-CAM shows the spatial regions that contributed most to the
model's activation for the predicted class. It does NOT prove
causality, and does NOT prove the model is "looking at the
defect" in any certified sense.

No pixel-level ground-truth defect masks exist for this dataset,
so the gate check (Phase 5.3) is a QUALITATIVE spatial-alignment
sanity check on a sample of true positives, not a quantitative
localization-accuracy metric.

## Gate verdict (human review)
**PASSED.** Reviewed a 10-sample true-positive grid. Heat consistently concentrates at the outer rim edge (the physical defect region) across all samples. Hot-spot position varies per image rather than sitting in a fixed location, consistent with tracking the actual visible flaw rather than a fixed shortcut. Center bore hole and background stay cool throughout. This is a qualitative spatial-alignment check, not a quantitative localization metric or proof of causality - no pixel-level ground-truth masks exist for this dataset.