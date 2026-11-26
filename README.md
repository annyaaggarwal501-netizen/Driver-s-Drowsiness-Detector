# DrowsyNet: driver drowsiness detection

The whole project lives in one notebook: **[`DrowsyNet.ipynb`](DrowsyNet.ipynb)**. It downloads the data, extracts
facial features, trains and compares 30+ models on people never seen in training, picks the best model with a
written rule, and writes and launches a **Streamlit** app that uses only that model.

App pages: **live webcam** (with alarm), **upload video**, **upload photo**, **model comparison**, **how it works**.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install jupyter
jupyter notebook DrowsyNet.ipynb      # then: Run All
```

The first cell installs the remaining packages. Expensive steps are cached in `data/`, `models/` and `reports/`, so
a re-run continues where it stopped. The last section starts the app at http://localhost:8501. You can also start
it directly once the notebook has run:

```powershell
streamlit run app.py
```

`drowsy_core.py` (the shared pipeline) and `app.py` (the Streamlit app) are **written by the notebook** with
`%%writefile`. Edit the notebook, not these files.

## What is new compared with the reference paper

Reference: C. Yashwanth & J. S. Kirar, *Driver's Drowsiness Detection*, IEEE TENCON 2019.

| Aspect | Reference paper | DrowsyNet |
|---|---|---|
| Landmarks | dlib, 68 points | MediaPipe: 478 points, 52 blendshapes, head pose |
| Time | single frames | personal calibration, PERCLOS, blinks, long closures, yawns, nods |
| Evaluation | same person in training and testing | unseen test participants + 5-fold group CV |
| Models | 4 classifiers | paper baseline, 14 classic ML, 8 CNN/ViT, stacking, late fusion (+ temporal models with video data) |
| Robustness | blurred frames broke 3 of 10 subjects | quality gate, low-light enhancement, CNN branch |
| Score | product of averages, unbounded | bounded 0–1 score mapped to KSS 1–9 |
| Explainability | none | reasons, Grad-CAM, feature importance |
| Product | offline experiment | Streamlit app: webcam, video, photo |

## Data

- [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd): 41,793 face images from 28 participants, downloaded automatically with `kagglehub`.
- Optional: [UTA-RLDD](https://sites.google.com/view/utarldd/home) videos in `data/raw/rldd/` with `USE_RLDD = True`, to train the window (Group B) and sequence (Group D) models.

Research prototype, not a certified driver-safety device.
