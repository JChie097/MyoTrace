# MyoTrace — Passive, Non-Invasive Pain Detection for Caregivers

> **Turn any ambient camera into structured intelligence.**
> Continuous, on-device facial-expression monitoring that catches *silent suffering* — pain a patient can't or won't verbalize — without a single cloud API.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00ACC1?style=flat-square)
![offline-first](https://img.shields.io/badge/offline--first-16a34a?style=flat-square)

MyoTrace watches a patient's face through an ordinary webcam and, using Google's MediaPipe FaceLandmarker running **fully on-device**, detects facial markers of pain (brow furrowing, eye squinting, upper-lip raising). When a pain signature is detected, it raises a local alarm, logs the event to an audit trail, and — optionally — relays an alert to a nurse station over the LAN. No video ever leaves the device.

---

## Problem statement

Many of the people most likely to be in pain are the least able to tell anyone:

- **Non-verbal patients** — stroke survivors, people with dementia or cognitive decline, intubated or post-operative patients, young children, and people in palliative care.
- **Silent suffering** — patients who under-report or mask pain out of stoicism, confusion, or fear of being a burden.

Caregivers can't be in the room every second, and staffing shortages mean checks happen on a schedule, not when a patient needs them. Between those checks, pain can go unobserved for hours.

Existing solutions don't fit: wearable sensors are intrusive, and cloud-based AI raises privacy concerns and requires a reliable internet connection — which care homes and home caregivers often can't guarantee, and can't afford at scale.

**MyoTrace's answer:** a passive, non-invasive monitor that never touches the patient, never interrupts the room, and never sends a frame to the cloud. It just watches — and speaks up when it sees a problem.

---

## Target users

- **Family caregivers** monitoring a parent or relative at home, especially overnight or from another room.
- **Care homes & assisted living** — one screen can quietly watch a resident who can't press a call button.
- **Hospitals & hospices** — non-verbal, intubated, or sedated patients in recovery and palliative wards.
- **Clinicians & technicians** — the explainability overlay and per-signal readout let them *verify and trust* what the model sees.

---

## How it works

1. **Capture** — a background thread reads the webcam at full camera rate and streams it to the dashboard over a local MJPEG feed (no re-encoding jitter).
2. **Detect** — MediaPipe's FaceLandmarker (a TensorFlow Lite model, running on CPU, offline) extracts **52 blendshape coefficients** for the face each frame.
3. **Map to pain signals** — blendshapes are mapped to facial action units:
   - `browDown` → **Brow contraction (AU4)**
   - `eyeSquint` → **Eye squint (AU6/7)**
   - `mouthUpperUp` → **Upper-lip raise (AU10)**
   - `mouthSmile` → **Smile (AU12)**
4. **Auto-calibrate** — ~15 frames of a neutral expression establish each patient's resting baseline. Pain gates are then **auto-tuned** from that patient's resting variability (z-score style), so no manual tuning is needed and the system adapts to glasses, resting squints, and other individual traits.
5. **Classify with hysteresis** — pain latches when **any 2 of 3** pain signals (brow, eye, lip) cross their gates *and* the patient is not smiling, held for 3 consecutive frames. It releases only after 3 frames below a 50% margin, which filters out flicker and single-frame noise.
6. **Respond** — the alarm tone is synthesized on-device (no audio file, no network), the event is stamped into the audit log, and an optional LAN relay can POST to a nurse station.

The result is **explainable by design**: every signal is shown as a live bar against its gate, and an opt-in heatmap overlay draws the exact facial regions the model is reading.

---

## 🛠️ Technology Stack

### 🧠 On-Device AI Engine
MediaPipe FaceLandmarker (TensorFlow Lite) · blendshape facial-action modeling · NumPy

### 📹 Computer Vision & Video
OpenCV · local MJPEG stream (stdlib) · background capture thread

### 🖥️ Dashboard & Interface
Streamlit · live signal telemetry · Vega-Lite trend chart · CSV audit export

### 🛡️ Privacy & Offline
100% on-device inference · no cloud APIs · synthesized alarm · optional LAN relay

---

## Key features

- 🔒 **Fully offline** — model, video stream, and alarm all run locally. No cloud, no API keys, no recurring cost.
- 🧠 **On-device ML** — MediaPipe FaceLandmarker blendshapes (TFLite) on CPU.
- 🎯 **Per-patient auto-calibration** — gates tune themselves from each patient's resting face.
- 🧯 **Hysteresis + debounce** — 2-of-3 rule with enter/exit streaks prevents false alarms.
- 😊 **Emotion-aware** — distinguishes pain from smiling, frustration, sneer, and tiredness so a smile is never mistaken for suffering.
- 📊 **Live telemetry** — signal bars, pain-confidence meter, and a confidence-over-time trend chart.
- 💾 **Audit trail** — timestamped event log with one-click CSV export.
- 👁️ **Explainability overlay** — optional heatmap showing what the model sees (off by default for a clean caregiver view).
- 📹 **Webcam or prerecorded clip** — demo from a live camera or a video file.
- 🎚️ **Sensitivity presets** — Standard / More / Less, plus advanced per-signal sliders for technicians.

---

## Requirements

- Python **3.12 or 3.13** (both tested)
- A webcam (built-in or USB)
- The dependencies in `requirements.txt`
- The MediaPipe model `face_landmarker_v2_with_blendshapes.task` — **already included in this repo**, so no download is needed (keeps the tool fully offline).

No GPU and no internet connection are required to run the detector.

---

## Setup & run

```bash
# 1. Clone the repository
git clone https://github.com/JChie097/MyoTrace.git
cd MyoTrace

# 2. Create and activate a virtual environment
python -m venv .venv

#    Windows:
.venv\Scripts\activate
#    macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Windows only) apply the MediaPipe "free symbol" workaround
python fix_mediapipe_windows.py

# 5. Run the dashboard
streamlit run app.py
```

Then open the single browser tab that Streamlit opens (`http://localhost:8501`).

### ⚠️ Important: the Windows MediaPipe fix

Recent MediaPipe Windows wheels ship a `libmediapipe.dll` that doesn't export the C `free` symbol, which crashes the first FaceLandmarker call with:

```
AttributeError: function 'free' not found
```

`fix_mediapipe_windows.py` patches the binding to fall back to the Windows C runtime. **Run it once after every `pip install`** (it's idempotent — safe to re-run). This step is Windows-only; macOS and Linux are unaffected.

### ⚠️ One camera, one tab

On Windows the webcam is exclusive to a single process. Run **one** Streamlit instance and keep **one** browser tab open — refreshing or opening a second tab creates a second session that competes for the camera and can blank the feed.

### Running from a prerecorded clip (demo fallback)

Set the video path in the sidebar (**Camera Source → Video File**), or via an environment variable:

```bash
# Windows (PowerShell)
$env:MYOTRACE_VIDEO="path\to\clip.mp4"; streamlit run app.py
```

The clip loops automatically. (Supply your own file — `demo.mp4` is not included.)

### Headless mode (no browser)

```bash
python myotrace_classifier.py
```

Opens a plain OpenCV window with the same classifier. Press **`r`** to recalibrate, **`q`** to quit.

---

## Using the dashboard

| Control | What it does |
|---|---|
| **Enforce Offline Isolation Mode** | On (default): fully offline; the alarm plays on-device. Off: enables an optional LAN relay to a nurse-station endpoint. |
| **Camera Source** | Live webcam or a video file. |
| **Show model heatmap overlay** | Off (default) for the clean caregiver view; on to reveal the colored brow/eye/lip heatmap the model reads. |
| **Sensitivity level** | **Auto (per-patient)** tunes the pain gates from the patient's own baseline. Standard / More / Less are fixed presets; **Advanced tuning** exposes the raw sliders. |
| **Reset Baseline Calibration Profile** | Re-runs the ~15-frame neutral calibration (hold a relaxed face for a few seconds). |

**To see the alarm:** after calibration, furrow your brows and raise/snarl your upper lip (any two of brow/eye/lip) for 2–3 seconds. The status turns red, the siren fires, and the event is logged.

---

## Social impact statement

Pain is one of the most under-treated symptoms in care, and it disproportionately affects the people least able to speak up for themselves. MyoTrace exists to give those patients a voice — **empowering caregivers to see suffering they'd otherwise miss**.

- **Privacy by architecture** — video is processed and discarded on-device. Nothing is uploaded, so a patient's most vulnerable moments never leave the room.
- **Affordable by design** — commodity webcam hardware, an open on-device model, and no per-use cloud fees. The cost is a one-time camera, not a subscription.
- **Equitable access** — runs offline, so it works in homes and facilities with poor or no internet, and on hardware budgets that can't sustain cloud AI.
- **Trustworthy by transparency** — the explainability overlay and live signal readout let caregivers *see* the reasoning, turning a black-box detector into something a nurse can believe in.

MyoTrace isn't a substitute for human care — it's a tool that makes that care more attentive, more dignified, and more evenly distributed.

---

## Disclaimer

⚠️ **Research prototype for demonstration only.** MyoTrace is **not a medical device** and is **not for clinical or diagnostic use**. Facial expression is one signal among many, and pain is complex; this tool provides an *assistive cue* to prompt a human caregiver to check on the patient, never a diagnosis.

Known limitations: single front-facing face, reasonably front-lit and unobstructed, within a few metres of the camera.

---

## Project structure

```
app.py                                   # Streamlit dashboard (capture, detection, classifier, UI)
myotrace_classifier.py                   # Headless OpenCV version of the same classifier
face_landmarker_v2_with_blendshapes.task # MediaPipe on-device model (bundled)
fix_mediapipe_windows.py                 # One-time Windows "free symbol" workaround
requirements.txt                         # Pinned dependencies
```

---

## Built with

[Python](https://www.python.org/) · [Streamlit](https://streamlit.io/) · [OpenCV](https://opencv.org/) · [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) · [TensorFlow Lite](https://www.tensorflow.org/lite) · NumPy · Pandas
