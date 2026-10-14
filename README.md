# Casting Defect Detection

ResNet18 classifier that finds defects in metal casting images, with Grad-CAM to show where the model is looking.

## The Problem

In a casting factory, most parts are fine and only a few have defects like cracks, holes or chipped edges. A model trained on this kind of data can get high accuracy just by calling everything "normal", which is useless because the whole point is to catch the bad parts. Missing a defect is much worse than a false alarm, so recall on the defect class matters more than accuracy.

There was a second problem I found in the data itself. The Kaggle dataset comes with its own train and test folders, but 64 images appear in both of them as exact copies, and many other images look almost the same. If one copy lands in train and another in test, the test score looks better than it really is because the model has basically seen the answer.

## My Approach

```
7,348 raw images
      |
      v
check for exact copies (MD5) and broken files
      |
      v
build a 4:1 normal-to-defect pool (3,921 images)
      |
      v
group near-duplicates (8x8 average hash)
      |
      v
split by group, stratified 70 / 15 / 15  --> no group in two splits
      |
      v
train ResNet18: plain CE (A) vs class-weighted CE (B)
      |
      v
pick threshold on validation set only
      |
      v
evaluate on test set once  -->  Grad-CAM + failure review
```

- **Imbalance** - the raw data actually has more defect images than normal ones, so I kept all 3,137 normal images and randomly sampled 784 defect images to get a 4:1 ratio (about 20% defects).
- **Leakage** - every image gets a perceptual hash. Images of the same class with the same hash go into one group, and the split is done on groups instead of single images. The code checks that no group ends up in two splits. This found 637 groups with more than one image (largest had 11).
- **Model** - ResNet18 with ImageNet weights. Only `layer4` and the final layer are trained, the rest is frozen. Adam (lr 3e-4), batch size 32, up to 30 epochs with early stopping on validation PR-AUC (patience 5). Light augmentation: flip, rotation up to 10 degrees, small brightness/contrast changes.
- **Loss comparison** - Experiment A uses normal cross-entropy, Experiment B weights the defect class higher based on train counts. The winner is picked by recall first, then PR-AUC, then F1.
- **Threshold** - instead of using 0.5, I tried thresholds from 0.1 to 0.8 on the validation set and took the one with the highest recall while keeping precision at least 0.7. The test set is only used once, at the end.

## Results

Test set: 568 images (445 normal, 123 defect)

| Metric | Model (B, threshold 0.10) | Always "normal" baseline |
|---|---|---|
| Recall | **1.000** | 0.000 |
| Precision | **0.976** | 0.000 |
| F1 | 0.988 | 0.000 |
| PR-AUC | **0.9997** | 0.2165 |
| Accuracy | 0.995 | 0.784 |

Confusion matrix: `[[442, 3], [0, 123]]` - every defect was caught, and 3 normal parts were flagged by mistake.

Experiment A vs B on validation:

| | Recall | PR-AUC | F1 |
|---|---|---|---|
| A - plain CE | 0.9916 | 0.9980 | 0.9874 |
| B - weighted CE | 0.9916 | 0.9984 | 0.9916 |

Both had the same recall, so B won on PR-AUC. Honestly the difference is small.

## Grad-CAM

I generated a heatmap for each test prediction using the last conv layer (`layer4.1.conv2`). When I looked at a sample of correctly caught defects, the heat was mostly on the outer rim where the defects are, and the hot spot moved around from image to image instead of staying in one fixed place. The center hole and background stayed cool.

This is only a visual check. The dataset has no pixel-level defect masks, so I can't measure how accurate the localization actually is.

## Failure Review

All 3 mistakes were normal parts predicted as defect. In each one the model focused on a mark on the rim that looks a lot like a chipped edge. Two of them had a defect probability of about 0.10, right on the threshold, and one was 0.82.

## Limitations

- The 4:1 imbalance was created by me by sampling, it is not the real factory ratio.
- The near-duplicate check only groups images with exactly the same hash, so copies that differ by a few bits are not caught.
- Grad-CAM shows where the model looks, not why it decided.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, Matplotlib, Streamlit

## Project Structure

```
casting_defect_detection/
|-- main.py                  runs the whole pipeline
|-- configs/config.yaml      all settings
|-- app/streamlit_app.py     demo app
|-- src/
|   |-- data/                image checks, imbalance, near-duplicates, split, dataset
|   |-- models/              ResNet18 setup and loss
|   |-- training/            training loop and A vs B selection
|   |-- evaluation/          threshold, metrics, baseline
|   |-- explainability/      Grad-CAM and failure review
|   |-- utils/               config, seed, paths
|-- data/
|   |-- raw/                 dataset goes here (not included)
|   |-- splits/              train / val / test CSV files
|-- outputs/                 checkpoints, figures, heatmaps, reports
```

## Run It

1. Download the [casting dataset from Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) and unzip it so you have `data/raw/casting_data/train/...` and `data/raw/casting_data/test/...`

2. Install and run:

```
pip install -r requirements.txt
python main.py
```

This runs all 6 phases and writes everything to `outputs/`. It takes around 15-30 minutes on my laptop (RTX 3050), most of it is training.

3. Try the demo:

```
streamlit run app/streamlit_app.py
```

Upload a casting image and it shows the prediction, the defect probability and the Grad-CAM overlay.
