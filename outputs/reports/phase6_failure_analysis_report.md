# Phase 6 Failure Analysis Report

- Total misclassifications: 3
- False positives (predicted defect, actually normal): 3
- False negatives (predicted normal, actually defect): 0

## Adaptation note
Only 3 misclassifications exist on the entire test set - too
few to populate all four failure-mode buckets with 3-5 representative
examples each, as the design doc's template assumes for a harder task.
Every failure is shown individually below instead of sampled.

## Failure-mode categories (design doc section 8)
- **A**: Background/acquisition artifact
- **B**: Correct region, weak defect
- **C**: Visual ambiguity (class confusion)
- **D**: Localization failure

## All misclassifications (grid): D:\Placements\ML_Projects\Cnn\outputs\figures\phase6_failure_grid.png
## Structured records (for categorization): D:\Placements\ML_Projects\Cnn\outputs\reports\phase6_failure_records.csv

## Categorization
Pending human visual review of the grid above. Each row in the
records CSV needs `observed_attention_region` and `failure_category`
filled in by hand once the images have been examined.