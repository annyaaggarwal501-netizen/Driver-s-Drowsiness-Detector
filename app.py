"""Driver Drowsiness Detection: live Streamlit app (Random Forest / KNN / CNN).

Online version (branch `online-deploy`): the "Live Camera (browser)" page streams the viewer's own
browser camera to the app with streamlit-webrtc, so it works when deployed on Streamlit Community Cloud.
The "Laptop Camera" page still uses OpenCV on the machine running the app (local runs only).

    pip install -r requirements.txt
    streamlit run app.py

The trained models come from DrowsyNet_3Models.ipynb and live in the models/ folder.
"""
import base64
import io
import json
import threading
import time
import urllib.request
import wave
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

MODELS = Path(__file__).parent / "models"
LANDMARKER = MODELS / "face_landmarker.task"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
INFO = json.loads((MODELS / "model_info.json").read_text())
FEATURES, IMG = INFO["features"], INFO["img_size"]
LABELS = INFO["labels"]                       # ["Non Drowsy", "Drowsy"]

# MediaPipe face-mesh points (same ones used to build the training data)
RIGHT_EYE, LEFT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS, MOUTH_VERTICAL = (61, 291), [(81, 178), (13, 14), (311, 402)]
FACE_TOP, FACE_BOTTOM = 10, 152
OUTLINES = {
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
}
BLENDSHAPES = {"eye_blink_left": "eyeBlinkLeft", "eye_blink_right": "eyeBlinkRight",
               "eye_squint_left": "eyeSquintLeft", "eye_squint_right": "eyeSquintRight",
               "gaze_down_left": "eyeLookDownLeft", "gaze_down_right": "eyeLookDownRight", "jaw_open": "jawOpen"}

ALARM_SECONDS = 1.5          # drowsy this long without a break -> alarm
SMOOTH_FRAMES = 10           # average the model output over the last 10 frames (~1 s)
TARGET_FPS = 10


# ----------------------------------------------------------------------------- CNN (same as the notebook)
class DrowsyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------- loading (cached once)
@st.cache_resource(show_spinner="Loading MediaPipe and the 3 models…")
def load_everything():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not LANDMARKER.exists():
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=LANDMARKER.read_bytes()),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    cnn = DrowsyCNN()
    cnn.load_state_dict(torch.load(MODELS / "cnn_model.pt", map_location="cpu"))
    cnn.eval()
    models = {"Random Forest": joblib.load(MODELS / "rf_model.pkl"),
              "KNN": joblib.load(MODELS / "knn_model.pkl"), "CNN": cnn}
    return mp, landmarker, models


mp, LANDMARKER_MODEL, MODEL_OBJECTS = load_everything()


@st.cache_resource
def detect_lock():
    return threading.Lock()


DETECT_LOCK = detect_lock()


# ----------------------------------------------------------------------------- face measurements (OpenCV + MediaPipe + NumPy)
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def enhance_if_dark(bgr):
    """Same low-light step as the training data: CLAHE when the image is dark."""
    if cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean() >= 60:
        return bgr
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def detect_face(bgr):
    """Returns (points [478, 2] in pixels, blendshapes dict, 4x4 matrix) or None."""
    h, w = bgr.shape[:2]
    for pad in (0, int(0.35 * max(h, w))):                 # retry with a border for tight face photos
        img = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with DETECT_LOCK:                                    # one detection at a time (several viewers online)
            res = LANDMARKER_MODEL.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.face_landmarks:
            ph, pw = img.shape[:2]
            pts = np.array([[p.x * pw - pad, p.y * ph - pad] for p in res.face_landmarks[0]], np.float32)
            blend = {c.category_name: c.score for c in res.face_blendshapes[0]}
            matrix = np.asarray(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
            return pts, blend, matrix
    return None


def dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    p = pts[idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-6)


def head_pose(m):
    if m is None:
        return 0.0, 0.0, 0.0
    r = m[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    pitch = np.arctan2(r[2, 1], r[2, 2]) if sy > 1e-6 else np.arctan2(-r[1, 2], r[1, 1])
    yaw = np.arctan2(-r[2, 0], sy)
    roll = np.arctan2(r[1, 0], r[0, 0]) if sy > 1e-6 else 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def measure(pts, blend, matrix):
    """The 14 features the Random Forest and KNN were trained on."""
    face_h = dist(pts[FACE_TOP], pts[FACE_BOTTOM]) + 1e-6
    pitch, yaw, roll = head_pose(matrix)
    f = {"ear_left": eye_aspect_ratio(pts, LEFT_EYE), "ear_right": eye_aspect_ratio(pts, RIGHT_EYE),
         "mar": sum(dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL) / (3 * dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]]) + 1e-6),
         "mouth_open": dist(pts[13], pts[14]) / face_h, "head_pitch": pitch, "head_yaw": yaw, "head_roll": roll}
    f.update({k: blend.get(v, 0.0) for k, v in BLENDSHAPES.items()})
    return f


def face_crop_gray(bgr, pts):
    """Square grayscale face crop resized to 64x64, framed like the dataset images
    (measured on the dataset: image side ≈ 1.06 × face size, centred 7.5 % of the face size above the face centre)."""
    h, w = bgr.shape[:2]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    size = max(x2 - x1, y2 - y1)
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2 - 0.075 * size, 1.06 * size / 2
    a, b, c, d = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    pad = max(0, -a, -b, c - w, d - h)
    img = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else bgr
    crop = img[b + pad:d + pad, a + pad:c + pad]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (IMG, IMG), interpolation=cv2.INTER_AREA)


def predict(model_name, feats, crop):
    """Probability that the driver is drowsy, from one of the three models."""
    if model_name == "CNN":
        x = torch.tensor(crop, dtype=torch.float32)[None, None] / 255.0
        with torch.no_grad():
            return float(torch.softmax(MODEL_OBJECTS["CNN"](x), 1)[0, 1])
    row = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
    return float(MODEL_OBJECTS[model_name].predict_proba(row)[0, 1])


def analyse(bgr):
    """Full pipeline for one frame: returns None if no face, else features, face crop and landmarks."""
    img = enhance_if_dark(bgr)
    found = detect_face(img)
    if found is None:
        return None
    pts, blend, matrix = found
    # the CNN was trained on the original (not brightened) images, so it gets a crop of the raw frame
    return {"feats": measure(pts, blend, matrix), "crop": face_crop_gray(bgr, pts), "pts": pts}


def draw(bgr, pts, color):
    out = bgr.copy()
    for idx in OUTLINES.values():
        cv2.polylines(out, [pts[idx].astype(np.int32)], True, color, 2, cv2.LINE_AA)
    return out


@st.cache_data
def beep_html():
    sr = 22050
    t = np.linspace(0, 0.4, int(sr * 0.4), False)
    tone = (np.sign(np.sin(2 * np.pi * 880 * t)) * 0.3 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(tone.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- page layout
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="😴", layout="wide")
st.markdown("""<style>
.status {padding:.6rem 1rem;border-radius:.7rem;color:white;font-size:1.4rem;font-weight:700;text-align:center}
.alarm {background:#dc2626;color:white;padding:.7rem;border-radius:.7rem;font-weight:800;text-align:center;
        font-size:1.3rem;animation:blink 1s infinite}
@keyframes blink {50% {opacity:.5}}
</style>""", unsafe_allow_html=True)

results = pd.DataFrame(INFO["results"]).T
with st.sidebar:
    st.title("😴 Drowsiness Detector")
    page = st.radio("Page", ["Live Camera (browser)", "Laptop Camera (local only)", "Upload Photo", "Model Comparison"])
    st.divider()
    names = list(results.index)
    model_name = st.selectbox("Model", names, index=names.index(INFO["best_model"]),
                              help="Switch models live to compare them")
    r = results.loc[model_name]
    st.caption(f"Test accuracy {r['Accuracy']:.1%} · F1 {r['F1-Score']:.1%} (on drivers never seen in training)")
    sound = st.toggle("Alarm sound", value=True)


# ----------------------------------------------------------------------------- Live camera in the browser (online)
def ice_servers():
    """STUN by default; a TURN server can be added in Streamlit secrets as ICE_SERVERS (for strict networks)."""
    try:
        if "ICE_SERVERS" in st.secrets:
            return json.loads(st.secrets["ICE_SERVERS"]) if isinstance(st.secrets["ICE_SERVERS"], str) else list(st.secrets["ICE_SERVERS"])
    except Exception:
        pass
    return [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]


class BrowserDetector:
    """Runs on every frame the browser sends; keeps the smoothing / alarm state for one viewer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model_name = INFO["best_model"]
        self.probs = deque(maxlen=SMOOTH_FRAMES)
        self.drowsy_since = self.closed_since = None
        self.info = {"face": False}

    def callback(self, frame):
        import av
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        now = time.time()
        res = analyse(img)
        if res is None:
            self.probs.clear()
            self.drowsy_since = self.closed_since = None
            cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2, cv2.LINE_AA)
            with self.lock:
                self.info = {"face": False}
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        with self.lock:
            name = self.model_name
        feats = res["feats"]
        self.probs.append(predict(name, feats, res["crop"]))
        p = float(np.mean(self.probs))
        drowsy = p >= 0.5
        self.drowsy_since = (self.drowsy_since or now) if drowsy else None
        eyes_closed = (feats["eye_blink_left"] + feats["eye_blink_right"]) / 2 > 0.55
        self.closed_since = (self.closed_since or now) if eyes_closed else None
        drowsy_for = now - self.drowsy_since if self.drowsy_since else 0.0
        alarm = drowsy_for >= ALARM_SECONDS
        color = (40, 40, 230) if drowsy else (60, 200, 60)
        vis = draw(img, res["pts"], color)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, f"{LABELS[int(drowsy)]}  {p:.0%}  [{name}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        if alarm:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 12)
            cv2.putText(vis, "WAKE UP!", (vis.shape[1] // 2 - 110, vis.shape[0] - 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        with self.lock:
            self.info = {"face": True, "p": p, "drowsy": drowsy, "alarm": alarm, "drowsy_for": drowsy_for,
                         "closed_for": (now - self.closed_since) if self.closed_since else 0.0,
                         "ear": (feats["ear_left"] + feats["ear_right"]) / 2, "mar": feats["mar"]}
        return av.VideoFrame.from_ndarray(vis, format="bgr24")


if page == "Live Camera (browser)":
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    st.header("Live Camera")
    st.caption("Uses **your browser's camera**, so it also works when the app runs online. Press **START** and allow "
               "camera access. Frames are analysed on the server and are not saved.")
    if "browser_detector" not in st.session_state:
        st.session_state.browser_detector = BrowserDetector()
    det = st.session_state.browser_detector
    with det.lock:
        det.model_name = model_name
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(key="drowsiness", mode=WebRtcMode.SENDRECV, video_frame_callback=det.callback,
                              rtc_configuration={"iceServers": ice_servers()},
                              media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                              async_processing=True)
        alarm_box, sound_box = st.empty(), st.empty()
    with right:
        status_box, prob_box = st.empty(), st.empty()
        m1, m2 = st.columns(2)
        ear_box, mar_box = m1.empty(), m2.empty()
        m3, m4 = st.columns(2)
        closed_box, drowsy_box = m3.empty(), m4.empty()
    if not ctx.state.playing:
        status_box.info("Press **START** under the video box to begin.")
    last_beep = 0.0
    while ctx.state.playing:
        with det.lock:
            info = dict(det.info)
        if not info.get("face"):
            status_box.markdown('<div class="status" style="background:#64748b">No face detected</div>', unsafe_allow_html=True)
            alarm_box.empty()
        else:
            label, bg = (LABELS[1], "#dc2626") if info["drowsy"] else (LABELS[0], "#16a34a")
            status_box.markdown(f'<div class="status" style="background:{bg}">{label}</div>', unsafe_allow_html=True)
            prob_box.progress(info["p"], text=f"Drowsiness probability: {info['p']:.0%}")
            ear_box.metric("Eye opening (EAR)", f"{info['ear']:.3f}")
            mar_box.metric("Mouth opening (MAR)", f"{info['mar']:.3f}")
            closed_box.metric("Eyes closed for", f"{info['closed_for']:.1f} s")
            drowsy_box.metric("Drowsy for", f"{info['drowsy_for']:.1f} s")
            if info["alarm"]:
                alarm_box.markdown('<div class="alarm">⚠ WAKE UP! Drowsiness detected</div>', unsafe_allow_html=True)
                if sound and time.time() - last_beep > 1.5:
                    last_beep = time.time()
                    sound_box.markdown(f'<audio autoplay src="data:audio/wav;base64,{beep_html()}"></audio><!--{last_beep}-->',
                                       unsafe_allow_html=True)
            else:
                alarm_box.empty()
        time.sleep(0.25)


# ----------------------------------------------------------------------------- Laptop camera (OpenCV, local runs only)
elif page == "Laptop Camera (local only)":
    st.header("Laptop Camera")
    st.caption("Opens the camera of the computer running this app with OpenCV. Use **Live Camera (browser)** when the app is online.")
    c1, c2 = st.columns([1, 3])
    cam_index = c1.number_input("Camera", 0, 5, 0, help="0 = built-in webcam")
    run = c2.toggle("▶ Start camera", key="run")
    left, right = st.columns([3, 2])
    frame_box, alarm_box, sound_box = left.empty(), left.empty(), left.empty()
    with right:
        status_box = st.empty()
        prob_box = st.empty()
        m1, m2 = st.columns(2)
        ear_box, mar_box = m1.empty(), m2.empty()
        m3, m4 = st.columns(2)
        closed_box, drowsy_box = m3.empty(), m4.empty()
        chart_box, fps_box = st.empty(), st.empty()

    if not run:
        frame_box.info("Press **Start camera**. Sit in front of the webcam with your face clearly visible.")
    else:
        cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Could not open the webcam. Close other apps using it (Zoom, Teams, Camera) or try camera 1.")
        else:
            probs, history = deque(maxlen=SMOOTH_FRAMES), deque(maxlen=150)
            drowsy_since = closed_since = None
            last, last_beep, t_start = 0.0, 0.0, time.time()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("The webcam stopped sending frames."); break
                    now = time.time()
                    if now - last < 1 / TARGET_FPS:
                        continue
                    t0, last = time.perf_counter(), now
                    frame = cv2.flip(frame, 1)
                    res = analyse(frame)
                    if res is None:
                        probs.clear(); drowsy_since = closed_since = None
                        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                        status_box.markdown('<div class="status" style="background:#64748b">No face detected</div>', unsafe_allow_html=True)
                        alarm_box.empty()
                        continue
                    feats = res["feats"]
                    probs.append(predict(model_name, feats, res["crop"]))
                    p = float(np.mean(probs))
                    drowsy = p >= 0.5
                    drowsy_since = (drowsy_since or now) if drowsy else None
                    eyes_closed = (feats["eye_blink_left"] + feats["eye_blink_right"]) / 2 > 0.55
                    closed_since = (closed_since or now) if eyes_closed else None
                    drowsy_for = now - drowsy_since if drowsy_since else 0.0
                    alarm = drowsy_for >= ALARM_SECONDS

                    color = (40, 40, 230) if drowsy else (60, 200, 60)
                    vis = draw(frame, res["pts"], color)
                    label = LABELS[int(drowsy)]
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
                    cv2.putText(vis, f"{label}  {p:.0%}  [{model_name}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    frame_box.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

                    bg = "#dc2626" if drowsy else "#16a34a"
                    status_box.markdown(f'<div class="status" style="background:{bg}">{label}</div>', unsafe_allow_html=True)
                    prob_box.progress(p, text=f"Drowsiness probability: {p:.0%}")
                    ear_box.metric("Eye opening (EAR)", f"{(feats['ear_left'] + feats['ear_right']) / 2:.3f}")
                    mar_box.metric("Mouth opening (MAR)", f"{feats['mar']:.3f}")
                    closed_box.metric("Eyes closed for", f"{(now - closed_since) if closed_since else 0:.1f} s")
                    drowsy_box.metric("Drowsy for", f"{drowsy_for:.1f} s")
                    if alarm:
                        alarm_box.markdown('<div class="alarm">⚠ WAKE UP! Drowsiness detected</div>', unsafe_allow_html=True)
                        if sound and now - last_beep > 1.5:
                            last_beep = now
                            sound_box.markdown(f'<audio autoplay src="data:audio/wav;base64,{beep_html()}"></audio><!--{now}-->',
                                               unsafe_allow_html=True)
                    else:
                        alarm_box.empty()
                    history.append({"time (s)": round(now - t_start, 1), "drowsiness": p})
                    if len(history) % 5 == 0:
                        chart_box.line_chart(pd.DataFrame(history).set_index("time (s)"), height=160)
                    fps_box.caption(f"{(time.perf_counter() - t0) * 1000:.0f} ms per frame")
            finally:
                cap.release()


# ----------------------------------------------------------------------------- Upload photo
elif page == "Upload Photo":
    st.header("Check a photo")
    up = st.file_uploader("Face photo", type=["jpg", "jpeg", "png", "bmp"])
    if up:
        bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        res = analyse(bgr)
        if res is None:
            st.error("No face found in this photo.")
        else:
            c1, c2 = st.columns([3, 2])
            c1.image(cv2.cvtColor(draw(bgr, res["pts"], (60, 200, 60)), cv2.COLOR_BGR2RGB), width="stretch")
            with c2:
                rows = []
                for name in results.index:
                    p = predict(name, res["feats"], res["crop"])
                    rows.append({"Model": name, "Prediction": LABELS[int(p >= 0.5)], "Drowsy probability": f"{p:.0%}"})
                st.subheader("All 3 models")
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                f = res["feats"]
                st.metric("Eye opening (EAR)", f"{(f['ear_left'] + f['ear_right']) / 2:.3f}")
                st.metric("Mouth opening (MAR)", f"{f['mar']:.3f}")
                st.image(res["crop"], caption="64×64 face image given to the CNN", width=160)


# ----------------------------------------------------------------------------- Model comparison
else:
    st.header("Model comparison")
    st.write("All three models were tested on the **same 6 drivers that were never used for training** "
             "(split by driver, so the models cannot memorise faces).")
    best = INFO["best_model"]
    st.success(f"Best model: **{best}**, F1 {results.loc[best, 'F1-Score']:.1%}, accuracy {results.loc[best, 'Accuracy']:.1%}")
    st.dataframe(results.style.format("{:.2%}").highlight_max(color="#bbf7d0"))
    st.bar_chart(results, stack=False, y_label="score (0–1)", height=320)
    st.markdown("""
| Model | Input | How it decides |
|---|---|---|
| **Random Forest** | 14 MediaPipe measurements (EAR, MAR, head pose, eye blink / squint / gaze, jaw open) | many decision trees vote |
| **KNN** | the same 14 measurements, scaled | looks at the 25 most similar training frames |
| **CNN** | 64×64 grayscale face image | convolution layers learn the features from the pixels |
""")
