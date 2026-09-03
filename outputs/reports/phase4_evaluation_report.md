# Phase 4 Evaluation Report

**Locked model: Experiment B.**

## Threshold selection (VALIDATION only)
Rule: maximize defect recall subject to defect precision >= 0.7 (stated before inspecting the table).

| threshold | precision | recall |
|---|---|---|
| 0.10 | 0.9752 | 0.9916 |
| 0.20 | 0.9833 | 0.9916 |
| 0.30 | 0.9833 | 0.9916 |
| 0.40 | 0.9833 | 0.9916 |
| 0.50 | 0.9916 | 0.9916 |
| 0.60 | 1.0000 | 0.9832 |
| 0.70 | 1.0000 | 0.9832 |
| 0.80 | 1.0000 | 0.9832 |

**Chosen threshold: 0.10** (precision=0.9752, recall=0.9916)

## Test set (touched once, this evaluation)
- Precision: 0.9762
- Recall: 1.0000
- F1: 0.9880
- PR-AUC: 0.9997
- Accuracy (secondary): 0.9947
- Confusion matrix: [[442, 3], [0, 123]]

## Majority-class do-nothing baseline (test set)
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- PR-AUC: 0.2165
- Accuracy: 0.7835

## Frozen predictions
Saved to: D:\Placements\ML_Projects\Cnn\outputs\predictions\test_predictions.csv
Phase 5/6 read this file post-hoc and must not alter it.