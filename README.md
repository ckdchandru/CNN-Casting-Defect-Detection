# Interpretable Defect Detection: CNN + Grad-CAM

CNN-based **casting defect detection** using ResNet18 transfer learning, class-imbalance handling, and Grad-CAM explainability.

### Method

```text
Dataset Verification
        ↓
Duplicate Detection
        ↓
Group-Aware Split
        ↓
Image Preprocessing
        ↓
ResNet18 Training
        ↓
Threshold Selection
        ↓
Test Evaluation
        ↓
Grad-CAM + Failure Analysis
```

### Dataset

Casting Product Image Data for Quality Inspection (Kaggle).

* 7,348 original images
* 3,137 OK images
* 784 defect images used for the imbalanced experiment
* Duplicate and near-duplicate checks performed
* Dataset not included in the repository

### Run

```bash
pip install -r requirements.txt
python main.py
```
