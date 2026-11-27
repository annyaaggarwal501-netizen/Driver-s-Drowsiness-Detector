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

## Run the live app (on your laptop, needs a webcam)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. On the **Live Webcam** page:
1. Choose a model in the sidebar (Random Forest, KNN or CNN). You can switch while the camera is on.
2. Press **Start camera**.
3. The app draws eye/mouth outlines and shows EAR, MAR and the drowsiness probability. If the model says Drowsy for
   1.5 seconds without a break, a **WAKE UP** alarm appears with a beep.

## How it works

1. **OpenCV** reads a webcam frame.
2. **MediaPipe Face Landmarker** finds 478 face points, eye-blink / squint / gaze / jaw scores and the head pose.
3. **NumPy** computes the eye aspect ratio, EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖), and the mouth aspect ratio, MAR.
4. **Random Forest / KNN** use 14 such measurements. The **CNN** uses the 64×64 grayscale face image.
5. The prediction is averaged over about 1 second. Staying drowsy for 1.5 s triggers the alarm.

Data source: Kaggle [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd).
