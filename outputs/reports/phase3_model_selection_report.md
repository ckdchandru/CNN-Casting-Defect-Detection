# Phase 3 Model Selection Report

**Winner: Experiment B.** pr_auc: A=0.9980 vs B=0.9984

## Experiment A (naive, unweighted CE)
- Best epoch: 11 (stopped early: True)
- Recall: 0.9916
- PR-AUC: 0.9980
- F1: 0.9874
- Precision: 0.9833

## Experiment B (weighted CE, PRIMARY)
- Best epoch: 13 (stopped early: True)
- Recall: 0.9916
- PR-AUC: 0.9984
- F1: 0.9916
- Precision: 0.9916

## Selection rule (locked, design doc section 9)
PRIMARY = defect recall, SECONDARY = defect PR-AUC, TERTIARY = defect F1.
Both experiments reported in full regardless of outcome.