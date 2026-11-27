# Driver Drowsiness Detection

This project detects whether a driver is **Drowsy** or **Alert** from their face. It uses **OpenCV** (webcam),
**MediaPipe** (face landmarks), **NumPy** (eye and mouth measurements), **scikit-learn** (Random Forest, KNN) and
**PyTorch** (CNN), and runs live in a **Streamlit** app.

## Files

| File | What it is |
|---|---|
| `DrowsyNet_3Models.ipynb` | Main notebook (Google Colab): Random Forest vs KNN vs CNN, trained and compared |
| `app.py` | Streamlit app: live webcam, photo check, model comparison |
| `models/` | Trained models from the notebook (`rf_model.pkl`, `knn_model.pkl`, `cnn_model.pt`, `model_info.json`) |
| `data/driver_drowsiness_features.csv` | Facial measurements of 41,793 face images (28 drivers) |
| `DrowsyNet_ML.ipynb` | Extended version: Logistic Regression, Random Forest, XGBoost, LSTM with personal calibration |

## Results (6 drivers never seen in training)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 69.14 % | 66.14 % | 72.01 % | 68.95 % |
| **KNN (best)** | **74.27 %** | 69.42 % | 82.05 % | **75.21 %** |
| CNN | 58.42 % | 53.68 % | 91.93 % | 67.78 % |

The data is split **by driver**, so no driver appears in both training and test. A random split would put
near-identical frames of the same person in both sets and give fake ~99 % accuracy.

## Run the notebook (Google Colab)

Open `DrowsyNet_3Models.ipynb` in Colab, choose Runtime → Change runtime type → **T4 GPU**, then **Run all**.
The CSV is read straight from this repository, and the face images are downloaded from Kaggle with `kagglehub`.

## Online version (this branch: `online-deploy`)

On this branch, the **Live Camera (browser)** page streams the viewer's own browser camera to the app with
`streamlit-webrtc`, so the app works when it is hosted online. The `main` branch keeps the laptop-only version.

### Deploy on Streamlit Community Cloud (free)
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `annyaaggarwal501-netizen/Driver-s-Drowsiness-Detector`, Branch: **`online-deploy`**,
   Main file path: `app.py`.
4. Optional: under **Advanced settings**, choose Python **3.12**.
5. Click **Deploy** and wait a few minutes for the first build (it installs MediaPipe, OpenCV and ONNX Runtime).
6. Share the link, e.g. `https://<your-name>.streamlit.app`. Open it, go to **Live Camera (browser)**, press **START**
   and allow camera access.

`requirements.txt` and `packages.txt` (system libraries) are kept small for the cloud. The online app does not
install PyTorch: the CNN trained in PyTorch is exported to `models/cnn_model.onnx` and run with ONNX Runtime, which
gives the same predictions. The headless OpenCV build is used because the server has no graphics libraries.

**Why the link is sometimes slow to open:** free Streamlit apps go to sleep when nobody uses them. The first visit
after that restarts the app, which reinstalls the packages and loads the models (a few minutes). Open the link a few
minutes before a demo.

**If the video never connects** (some college or office networks block it), add a TURN relay server in the app's
**Settings → Secrets** on Streamlit Cloud, for example from a free metered.ca or Twilio account:

```toml
ICE_SERVERS = '[{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["turn:YOUR_TURN_HOST:443"], "username": "USER", "credential": "PASSWORD"}]'
```

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. On either camera page:
1. Choose a model in the sidebar (Random Forest, KNN or CNN). You can switch while the camera is on.
2. Press **START** (browser camera) or **Start camera** (laptop camera).
3. The app draws eye/mouth outlines and shows EAR, MAR and the drowsiness probability. If the model says Drowsy for
   1.5 seconds without a break, a **WAKE UP** alarm appears with a beep.

## How it works

1. **OpenCV** reads a webcam frame.
2. **MediaPipe Face Landmarker** finds 478 face points, eye-blink / squint / gaze / jaw scores and the head pose.
3. **NumPy** computes the eye aspect ratio, EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖), and the mouth aspect ratio, MAR.
4. **Random Forest / KNN** use 14 such measurements. The **CNN** uses the 64×64 grayscale face image.
5. The prediction is averaged over about 1 second. Staying drowsy for 1.5 s triggers the alarm.

Data source: Kaggle [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd).
