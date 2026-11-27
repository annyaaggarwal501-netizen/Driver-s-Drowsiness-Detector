# DrowsyNet: Driver Drowsiness Detection with Machine Learning

The whole project is one Google Colab notebook: **[`DrowsyNet_ML.ipynb`](DrowsyNet_ML.ipynb)**.

It classifies each video frame of a driver's face as **Drowsy** or **Non Drowsy** from facial measurements (eye
aspect ratio, mouth aspect ratio, head pose, eye blink / squint / gaze, jaw opening). It compares 4 models on
**drivers never seen during training**, picks the best one automatically, saves it, and provides a
`predict_drowsiness()` function.

## Results (6 unseen test drivers)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 57.90 % | 79.44 % | 34.78 % | 48.38 % | 0.715 |
| Random Forest | 88.80 % | 96.82 % | 82.98 % | 89.37 % | 0.961 |
| **XGBoost (best)** | **90.34 %** | **97.99 %** | **84.69 %** | **90.86 %** | **0.956** |
| LSTM (30-frame sequences) | 82.09 % | 90.55 % | 76.39 % | 82.87 % | 0.862 |

## How to run (Google Colab)

1. Open `DrowsyNet_ML.ipynb` in Colab (File → Open notebook → GitHub → paste this repository's URL).
2. Download [`data/driver_drowsiness_features.csv`](data/driver_drowsiness_features.csv) from this repository.
3. Runtime → Run all. When the file picker appears, upload the CSV.
4. The best model, preprocessor and label encoder are saved and downloaded at the end. Training takes a few minutes.

## Dataset

`data/driver_drowsiness_features.csv` has 41,793 rows (one per frame) from 28 drivers. The measurements were
extracted with Google MediaPipe Face Landmarker from the face images of the Kaggle
[Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd),
which comes from the UTA Real-Life Drowsiness Dataset videos.

Columns: `image`, `subject_id`, `frame`, `label` (Drowsy / Non Drowsy) and 25 measurements: EAR left/right, MAR,
mouth opening, head pitch/yaw/roll, eye blink/squint/wide, gaze down/up, jaw open, mouth close, brow movements,
face scale, image blur and brightness.

## Key methods

- **No data leakage.** The split is by driver (`GroupShuffleSplit` on `subject_id`), so no driver appears in both
  training and test. Time-based features and LSTM sequences never cross drivers or recordings.
- **Feature engineering.** Derived eye features, rolling PERCLOS-style features over the last 30 frames, and
  **personal calibration**: z-scores against each driver's own alert baseline. Calibration is the largest single
  improvement, because differences between people are bigger than the drowsy/alert difference.
- **Best-model rule.** F1 first, then recall for Drowsy, then cross-validated F1 across drivers, then accuracy.

Research prototype, not a certified driver-safety device.
