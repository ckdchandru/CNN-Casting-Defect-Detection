<<<<<<< HEAD
# Interpretable Defect Detection: CNN + Grad-CAM

A CNN-based image classification project for detecting casting defects, with class-imbalance handling and Grad-CAM-based model interpretation.

## Problem

The project focuses on:

- Detecting defective castings from top-view images.
- Handling class imbalance during training.
- Using Grad-CAM to visualize the regions influencing predictions.
- Analyzing model failures.

## Dataset

**Casting Product Image Data for Quality Inspection** (Kaggle).

The original dataset contains 7,348 images:

- `def_front`: 4,211
- `ok_front`: 3,137

To create a defect-minority setting, all 3,137 `ok_front` images were retained and `def_front` was randomly reduced to 784 images using seed 42.

The dataset was also checked for duplicate and near-duplicate images before creating the train/validation/test split. Related images were grouped to prevent data leakage between splits.

The dataset is not included in this repository.

## Method

```text
Dataset Verification
        ↓
Duplicate Detection
        ↓
Group-Aware Train/Val/Test Split
        ↓
Image Preprocessing
        ↓
ResNet18 Training
        ↓
Threshold Selection
        ↓
Test Evaluation
        ↓
Grad-CAM
        ↓
Failure Analysis
=======
# interpretable-defect-detection-cnn
Industrial defect detection using ResNet18 transfer learning and Grad-CAM explainability with PyTorch.
>>>>>>> 97724940911e69cd28beab4c80e14de22ce92e1b
