# Phase 6 Failure Analysis Report

- Total misclassifications: 3
- False positives (predicted defect, actually normal): 3
- False negatives (predicted normal, actually defect): 0

## Adaptation note
Only 3 misclassifications exist on the entire test set - too
few to fill all four failure-mode buckets with 3-5 examples each.
Every failure is shown individually below instead of sampled.

## Failure-mode categories
- **A**: Background/acquisition artifact
- **B**: Correct region, weak defect
- **C**: Visual ambiguity (class confusion)
- **D**: Localization failure

## All misclassifications (grid): outputs\figures\phase6_failure_grid.png
## Structured records (for categorization): outputs\reports\phase6_failure_records.csv

## Categorization
Each row in the records CSV needs `observed_attention_region` and
`failure_category` filled in by hand after looking at the grid.