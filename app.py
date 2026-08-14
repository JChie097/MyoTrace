import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import csv
import datetime
import io
import json
import re
import time
import threading
import http.server
import socketserver
import urllib.request


def event_logs_to_csv(entries):
    """Convert the HTML audit log entries into a two-column CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "event"])
    for entry in entries:
        text = re.sub(r"<[^>]+>", "", entry).strip()
        m = re.search(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)", text)
        if m:
            writer.writerow([m.group(1), m.group(2)])
        else:
            writer.writerow(["", text])
    # UTF-8 BOM so Excel/Sheets detect the encoding and render emoji correctly.
    return "\ufeff" + buf.getvalue()


def alarm_tone(sample_rate=44100, duration=0.7):
    """Generate an on-device siren tone (16-bit PCM). No files, no network."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 880.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t))
    fade = max(1, int(0.02 * sample_rate))
    env = np.ones_like(tone)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (tone * env * 32767).astype(np.int16)


def relay_alert(message, url):
    """POST a pain alert to a LAN nurse-station endpoint. Returns (ok, detail)."""
    try:
        payload = json.dumps({
            "event": "pain_alert",
            "message": message,
            "timestamp": datetime.datetime.now().isoformat(),
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True, f"relayed to nurse station (HTTP {resp.status})"
    except Exception as e:
        return False, f"relay failed: {e}"


def blendshape_pair(scores, names):
    """Average a left/right blendshape pair into one 0..1 activation value."""
    return (scores.get(names[0], 0.0) + scores.get(names[1], 0.0)) / 2.0


# 1. PREMIUM LAYOUT & VISUAL STYLING INJECTION
st.set_page_config(page_title="MyoTrace Caregiver Portal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main { background-color: #0b0f19; color: #f8fafc; }
        h1 { color: #ffffff; font-weight: 800; font-size: 2.5rem !important; letter-spacing: -0.05rem; }
        h2 { color: #cbd5e1; font-weight: 600; font-size: 1.4rem !important; }
        .stButton>button {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            color: white !important; border-radius: 8px !important; border: none !important;
            padding: 0.6rem 1.8rem !important; font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); transition: all 0.2s ease-in-out; width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
            background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
        }
        .metric-card { background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { background-color: #111827; border: 1px solid #1f2937 !important; border-radius: 12px; }
        h3 { color: #cbd5e1; font-weight: 600; }
        .brand-badge { background-color: #1e293b; color: #38bdf8; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-bottom: 1rem; }
        .log-box { background-color: #030712; border: 1px solid #374151; font-family: monospace; padding: 1rem; border-radius: 8px; height: 180px; overflow-y: scroll; color: #34d399; font-size: 0.9rem; }
        .feed-img { width: 100%; height: auto; display: block; border-radius: 8px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)


# 2. LOCAL MJPEG VIDEO STREAM (stdlib only, fully offline).
# The browser <img> reads this URL directly, so the video feed is never re-rendered
# through Streamlit — this is what eliminates the frame flicker.
@st.cache_resource
def start_mjpeg_stream():
    stream = {"lock": threading.Lock(), "frame": b"", "port": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/stream.mjpg":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            try:
                while True:
                    with stream["lock"]:
                        jpeg = stream["frame"]
                    if jpeg:
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    time.sleep(0.04)
            except Exception:
                pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    base_port = int(os.environ.get("MYOTRACE_MJPEG_PORT", "8765"))
    for port in range(base_port, base_port + 30):
        try:
            server = Server(("127.0.0.1", port), Handler)
        except OSError:
            continue
        stream["port"] = port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        break

    return stream


STREAM = start_mjpeg_stream()
stream_url = f"http://127.0.0.1:{STREAM['port']}/stream.mjpg"


class CameraEngine:
    """Captures frames in a daemon thread at full camera rate and publishes them to
    the MJPEG stream. MediaPipe detection runs in parallel (in the fragment) on the
    latest captured frame. Sharing one engine via st.cache_resource means every
    browser tab reuses a single camera handle — which also fixes the Windows
    "camera is in use by another tab" contention problem."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self.source_key = None
        self.is_webcam = True
        self.frame_rgb = None   # latest RGB frame for MediaPipe
        self.frame_id = 0       # increments on every captured frame
        self.error = None       # no_camera / video_open_fail / video_read_fail / not_delivering
        self.fail_count = 0     # consecutive failed reads (transient)
        self.running = False
        self.overlay = None     # heatmap regions+colors, published by the classifier fragment

    def start(self, source_key, is_webcam):
        if self.source_key == source_key and self.running:
            return
        self.stop()
        self.source_key = source_key
        self.is_webcam = is_webcam
        self.error = None
        self.fail_count = 0
        self.frame_rgb = None
        self.frame_id = 0

        if source_key.startswith("video:"):
            cap = cv2.VideoCapture(source_key[len("video:"):])
            if not cap.isOpened():
                self.error = "video_open_fail"
                return
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.error = "no_camera"
                return

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="myotrace-capture")
        self.running = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.running = False

    def latest(self):
        with self._lock:
            return self.frame_rgb, self.frame_id, self.error, self.fail_count

    def set_overlay(self, overlay):
        with self._lock:
            self.overlay = overlay

    def get_overlay(self):
        with self._lock:
            return self.overlay

    @staticmethod
    def _draw_overlay(bgr, overlay):
        """Draw semi-transparent heatmap boxes (points + RGB color) onto a BGR frame."""
        h, w = bgr.shape[:2]
        out = bgr.copy()
        for points, color_rgb in overlay:
            xs = [int(p[0] * w) for p in points]
            ys = [int(p[1] * h) for p in points]
            if not xs or not ys:
                continue
            x1, y1 = max(min(xs) - 3, 0), max(min(ys) - 3, 0)
            x2, y2 = min(max(xs) + 3, w - 1), min(max(ys) + 3, h - 1)
            r, g, b = color_rgb
            color_bgr = (int(b), int(g), int(r))
            layer = out.copy()
            cv2.rectangle(layer, (x1, y1), (x2, y2), color_bgr, thickness=-1)
            cv2.addWeighted(layer, 0.35, out, 0.65, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness=2)
        return out

    def _run(self):
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self.error = "no_camera" if self.is_webcam else "video_open_fail"
                break

            success, frame = self._cap.read()
            if not success:
                if not self.is_webcam:
                    # End of clip -> loop back to the start.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = self._cap.read()
                    if not success:
                        self.error = "video_read_fail"
                        break
                else:
                    self.fail_count += 1
                    if self.fail_count >= 10:
                        self.error = "not_delivering"
                        break
                    time.sleep(0.05)
                    continue
            else:
                self.fail_count = 0

            if self.is_webcam:
                frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw the pain-region heatmap (if the classifier has published one), then
            # publish to the MJPEG stream (imencode expects BGR, so keep BGR).
            overlay = self.get_overlay()
            display = self._draw_overlay(frame, overlay) if overlay else frame
            ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with STREAM["lock"]:
                    STREAM["frame"] = jpeg.tobytes()

            with self._lock:
                self.frame_rgb = frame_rgb
                self.frame_id += 1

            # Webcam read() already blocks at the camera's rate; pace video files to
            # their native FPS so prerecorded clips play back in real time.
            if not self.is_webcam:
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                time.sleep(1.0 / max(min(fps, 60.0), 1.0))

        self.running = False


@st.cache_resource
def get_camera_engine():
    return CameraEngine()

# Initialize Session States with safe array defaults
if 'base_brow' not in st.session_state: st.session_state.base_brow = None
if 'base_eye' not in st.session_state: st.session_state.base_eye = None
if 'base_lip' not in st.session_state: st.session_state.base_lip = None
if 'base_smile' not in st.session_state: st.session_state.base_smile = None
if 'calibrated' not in st.session_state: st.session_state.calibrated = False
if 'sigma_brow' not in st.session_state: st.session_state.sigma_brow = None
if 'sigma_eye' not in st.session_state: st.session_state.sigma_eye = None
if 'sigma_lip' not in st.session_state: st.session_state.sigma_lip = None
if 'auto_thresh_brow' not in st.session_state: st.session_state.auto_thresh_brow = None
if 'auto_thresh_eye' not in st.session_state: st.session_state.auto_thresh_eye = None
if 'auto_thresh_lip' not in st.session_state: st.session_state.auto_thresh_lip = None
if 'thresh_brow' not in st.session_state: st.session_state.thresh_brow = 12
if 'thresh_eye' not in st.session_state: st.session_state.thresh_eye = 10
if 'thresh_lip' not in st.session_state: st.session_state.thresh_lip = 8
if 'thresh_smile' not in st.session_state: st.session_state.thresh_smile = 20
if 'history' not in st.session_state: st.session_state.history = [0] * 30
if 'alarm_active' not in st.session_state: st.session_state.alarm_active = False
if 'smoothed_brow' not in st.session_state: st.session_state.smoothed_brow = 0.0
if 'smoothed_eye' not in st.session_state: st.session_state.smoothed_eye = 0.0
if 'smoothed_lip' not in st.session_state: st.session_state.smoothed_lip = 0.0
if 'smoothed_smile' not in st.session_state: st.session_state.smoothed_smile = 0.0
if 'pain_active' not in st.session_state: st.session_state.pain_active = False
if 'pain_streak' not in st.session_state: st.session_state.pain_streak = 0
if 'calib_buffer' not in st.session_state: st.session_state.calib_buffer = []
if 'csv_cache' not in st.session_state: st.session_state.csv_cache = ""
if 'csv_log_count' not in st.session_state: st.session_state.csv_log_count = -1
if 'log_render_count' not in st.session_state: st.session_state.log_render_count = -1
if 'log_html' not in st.session_state: st.session_state.log_html = ""
if 'event_logs' not in st.session_state:
    st.session_state.event_logs = [
        '<div style="color: #6b7280; margin-bottom: 4px;">🕒 System initialized. Awaiting baseline configuration...</div>']

# 3. SIDEBAR SYSTEM SETTINGS PANEL
GATE_KEYS = ("thresh_brow", "thresh_eye", "thresh_lip", "thresh_smile")
AUTO_LABEL = "Auto (per-patient)"
PRESETS = {
    "Standard (recommended)": (12, 10, 8, 20),
    "More sensitive": (8, 6, 5, 15),
    "Less sensitive": (20, 18, 14, 30),
}
SENSITIVITY_OPTIONS = [AUTO_LABEL] + list(PRESETS.keys())

with st.sidebar:
    st.header("⚙️ Core Controls")
    network_mode = st.toggle("Enforce Offline Isolation Mode", value=True, key="network_mode")
    if network_mode:
        st.success("🔒 Offline isolation enforced — no outbound connections; alarm plays on-device.")
    else:
        st.warning("⚠️ Intranet mode — alerts may be relayed to a nurse station on the LAN.")
        st.text_input(
            "Nurse Station Alert Endpoint",
            value="http://nurse-station:8000/alert",
            help="LAN URL that receives a JSON POST each time a pain alert fires.",
            key="relay_url",
        )

    st.markdown("---")
    st.subheader("📹 Camera Source")
    camera_source = st.radio(
        "Input source",
        ["Live Webcam", "Video File"],
        index=(1 if os.environ.get("MYOTRACE_VIDEO") else 0),
        key="camera_source",
    )
    if camera_source == "Video File":
        st.text_input(
            "Video file path",
            value=os.environ.get("MYOTRACE_VIDEO", "demo.mp4"),
            help="Path to a prerecorded clip on the machine running this app (also settable via the MYOTRACE_VIDEO env var).",
            key="video_path",
        )

    st.markdown("---")
    st.subheader("👁️ Feed Display")
    show_heatmap = st.toggle(
        "Show model heatmap overlay",
        value=False,
        key="show_heatmap",
        help="Overlay the live pain-signal heatmap on the feed (what the model sees). Off by default for the clean caregiver view.",
    )

    st.markdown("---")
    st.subheader("🎛️ Alert Sensitivity")

    preset = st.radio(
        "Sensitivity level",
        SENSITIVITY_OPTIONS,
        index=0,
        key="sensitivity_preset",
        help="Auto tunes the pain gates to this patient's resting face during calibration, so no manual setting is needed. Choose a fixed preset only if you want manual control.",
    )

    if st.session_state.get("applied_preset") != preset:
        if preset in PRESETS:
            for key, val in zip(GATE_KEYS, PRESETS[preset]):
                st.session_state[key] = val
        st.session_state.applied_preset = preset

    auto_mode = (preset == AUTO_LABEL)
    if auto_mode:
        st.caption("Pain gates are auto-tuned from this patient's calm baseline during calibration — no manual setting needed.")
    else:
        st.caption("Pain alert fires when any 2 of brow, eye squint, and lip raise cross their gates.")

    with st.expander("Advanced tuning (for technicians)"):
        if auto_mode:
            st.caption("Sliders are disabled in Auto mode — gates are computed from the patient's resting baseline.")
        else:
            st.caption("Lower a gate = more sensitive to that expression.")
        # No `value=` here: the preset application above seeds session_state,
        # and passing both a default and a session-state value triggers a Streamlit warning.
        brow_val = st.slider("Brow Contraction Gate (AU4)", 5, 50, key="thresh_brow", disabled=auto_mode)
        eye_val = st.slider("Eyelid Squint Gate (AU7)", 5, 50, key="thresh_eye", disabled=auto_mode)
        lip_val = st.slider("Upper Lip Raise Gate (AU10)", 5, 50, key="thresh_lip", disabled=auto_mode)
        smile_val = st.slider("Smile Gate (AU12)", 5, 50, key="thresh_smile", disabled=auto_mode)
        if not auto_mode and (brow_val, eye_val, lip_val, smile_val) != PRESETS[preset]:
            st.caption("⚠️ Custom values in use — advanced sliders override the preset.")

# Log network-mode transitions so isolation changes are auditable.
if 'last_network_mode' not in st.session_state:
    st.session_state.last_network_mode = network_mode
if network_mode != st.session_state.last_network_mode:
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    if network_mode:
        st.session_state.event_logs.append(
            f'<div style="color: #3b82f6; margin-bottom: 4px;">🔒 [{stamp}] Offline isolation enforced — network relay disabled.</div>')
    else:
        st.session_state.event_logs.append(
            f'<div style="color: #f59e0b; margin-bottom: 4px;">🌐 [{stamp}] Intranet mode enabled — alert relay armed.</div>')
    st.session_state.last_network_mode = network_mode

# 4. FACS-TO-BLENDSHAPE SIGNAL MAPPING (MediaPipe FaceBlendshapes)
BLENDSHAPE_BROW = ('browDownLeft', 'browDownRight')
BLENDSHAPE_SQUINT = ('eyeSquintLeft', 'eyeSquintRight')
BLENDSHAPE_LIP = ('mouthUpperUpLeft', 'mouthUpperUpRight')
BLENDSHAPE_SMILE = ('mouthSmileLeft', 'mouthSmileRight')

SMOOTH_ALPHA = 0.35       # EMA smoothing factor (higher = more responsive)
CALIB_FRAMES = 15         # frames averaged for the neutral baseline
PAIN_ENTER_STREAK = 3     # consecutive frames required to raise the alert
PAIN_EXIT_STREAK = 3      # consecutive frames required to clear the alert
HYSTERESIS_MARGIN = 0.5   # exit threshold = 50% of the enter threshold
PAIN_SMILE_GUARD = 20     # a smile above this percent blocks pain (decoupled from the smile gate)
HISTORY_MAX = 300         # confidence history length kept for the trend chart
AUTO_SIGMA_K = 4.0        # auto pain gate = k × the patient's resting std-dev (z-score style)
AUTO_FLOOR = 6            # minimum auto gate (%) so a perfectly still face isn't hyper-sensitive
AUTO_CEIL = 35            # maximum auto gate (%) so a fidgety baseline can't desensitize the alert


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def auto_gate(sigma):
    """Map a resting-blendshape std-dev to a pain gate (percent above baseline)."""
    if sigma is None:
        return None
    return round(min(max(AUTO_SIGMA_K * sigma * 100.0, AUTO_FLOOR), AUTO_CEIL), 1)


# FaceMesh landmark indices (0..467) used to draw the pain-region heatmap bands.
HEAT_BROW = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]            # both eyebrows
HEAT_EYE = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]  # both eyes
HEAT_LIP = [61, 291, 13, 14, 0, 81, 311]                                # mouth / upper lip


def heat_color(pct):
    """Map a signal % to a heat color: 0% green, 50% yellow, 100% red."""
    p = min(max(pct, 0.0), 100.0)
    if p < 50.0:
        t = p / 50.0
        return (int(34 + (250 - 34) * t), int(197 + (204 - 197) * t), int(94 + (21 - 94) * t))
    t = (p - 50.0) / 50.0
    return (int(250 + (239 - 250) * t), int(204 + (68 - 204) * t), int(21 + (68 - 21) * t))


def region_points(landmarks, indices):
    """Return normalized (x, y) points for a set of landmark indices."""
    return [(landmarks[i].x, landmarks[i].y) for i in indices]

# 5. FACE LANDMARKER (MediaPipe on-device TFLite model)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
)

# 6. USER INTERFACE HEADER DESIGN (rendered once; never part of the live loop)
st.markdown('<div class="brand-badge">• Powered by Edge-Native Local AI Engine</div>', unsafe_allow_html=True)
st.title("Turn any ambient camera into structured intelligence")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -1rem; margin-bottom: 2rem;'>Passive, continuous myo-spatial landmark tracking for non-verbal caregiver assistance.</p>",
    unsafe_allow_html=True)
st.caption("⚠️ Research prototype for demonstration only — not a medical device and not for clinical or diagnostic use.")


# 7. LIVE DASHBOARD — a fragment that re-runs only its own section on a timer.
#    Camera frames are captured by a background thread (CameraEngine) at full rate and
#    published to the MJPEG <img> directly, while this fragment runs MediaPipe
#    detection in parallel and updates the telemetry widgets. No whole-page re-render,
#    no frame flicker, and smooth full-rate video.
@st.fragment(run_every=0.18)
def live_dashboard():
    # Read control values from session_state (set by the sidebar widgets above).
    network_mode = st.session_state.get("network_mode", True)
    relay_url = st.session_state.get("relay_url", "")
    camera_source = st.session_state.get("camera_source", "Live Webcam")
    video_path = st.session_state.get("video_path", os.environ.get("MYOTRACE_VIDEO", "demo.mp4"))
    show_heatmap = st.session_state.get("show_heatmap", False)
    thresh_brow = st.session_state.get("thresh_brow", 12)
    thresh_eye = st.session_state.get("thresh_eye", 10)
    thresh_lip = st.session_state.get("thresh_lip", 8)
    thresh_smile = st.session_state.get("thresh_smile", 20)

    # Effective pain gates: in Auto mode they come from the patient's own resting
    # baseline (falling back to the Standard values until calibration completes).
    sensitivity = st.session_state.get("sensitivity_preset", AUTO_LABEL)
    if sensitivity == AUTO_LABEL:
        eff_brow = st.session_state.get("auto_thresh_brow") or thresh_brow
        eff_eye = st.session_state.get("auto_thresh_eye") or thresh_eye
        eff_lip = st.session_state.get("auto_thresh_lip") or thresh_lip
    else:
        eff_brow, eff_eye, eff_lip = thresh_brow, thresh_eye, thresh_lip

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("📹 Live Patient Room Feed")
            if STREAM["port"]:
                # Constant markup -> Streamlit keeps the same <img> node, so the browser
                # streams MJPEG continuously without a single re-render/flicker.
                st.markdown(
                    f'<img class="feed-img" src="{stream_url}" alt="Live patient room feed">',
                    unsafe_allow_html=True)
                if show_heatmap:
                    st.caption("Heat overlay — 🟢 calm · 🟡 raised · 🔴 pain (brow / eyes / upper lip)")
            else:
                st.warning("📹 Could not start the local video stream — restart the app.")
            if st.button("Reset Baseline Calibration Profile"):
                st.session_state.calibrated = False
                st.session_state.base_brow, st.session_state.base_eye, st.session_state.base_lip, st.session_state.base_smile = None, None, None, None
                st.session_state.sigma_brow = st.session_state.sigma_eye = st.session_state.sigma_lip = None
                st.session_state.auto_thresh_brow = st.session_state.auto_thresh_eye = st.session_state.auto_thresh_lip = None
                st.session_state.calib_buffer = []
                st.session_state.smoothed_brow = st.session_state.smoothed_eye = st.session_state.smoothed_lip = st.session_state.smoothed_smile = 0.0
                st.session_state.pain_active = False
                st.session_state.pain_streak = 0
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                st.session_state.event_logs.append(
                    f'<div style="color: #eab308; margin-bottom: 4px;">🔄 [{timestamp}] Calibration baseline profile wiped by operator.</div>')

    with col2:
        with st.container(border=True):
            st.subheader("📊 Caregiver Telemetry Alerts")
            status_box = st.container()
            chart_box = st.container()
            trend_box = st.container()
            audio_box = st.container()

    # 8. DYNAMIC EVENT LOGGER PANEL
    with st.container(border=True):
        st.subheader("💾 Real-Time Session Audit Trail Logs")
        log_container = st.container()
        if st.session_state.csv_log_count != len(st.session_state.event_logs):
            st.session_state.csv_cache = event_logs_to_csv(st.session_state.event_logs)
            st.session_state.csv_log_count = len(st.session_state.event_logs)
        st.download_button(
            "⬇️ Export Audit Trail (CSV)",
            data=st.session_state.csv_cache,
            file_name="myotrace_audit_trail.csv",
            mime="text/csv",
        )

    # 9. CAMERA (background thread) + LANDMARKER (on-device model)
    source_key = f"video:{video_path}" if camera_source == "Video File" else "webcam:0"
    is_webcam = camera_source == "Live Webcam"

    engine = get_camera_engine()
    if st.session_state.get('cap_source') != source_key:
        engine.start(source_key, is_webcam)
        st.session_state.cap_source = source_key
        # New source -> fresh calibration profile.
        st.session_state.calibrated = False
        st.session_state.base_brow = st.session_state.base_eye = st.session_state.base_lip = st.session_state.base_smile = None
        st.session_state.sigma_brow = st.session_state.sigma_eye = st.session_state.sigma_lip = None
        st.session_state.auto_thresh_brow = st.session_state.auto_thresh_eye = st.session_state.auto_thresh_lip = None
        st.session_state.calib_buffer = []
        st.session_state.smoothed_brow = st.session_state.smoothed_eye = st.session_state.smoothed_lip = st.session_state.smoothed_smile = 0.0
        st.session_state.pain_active = False
        st.session_state.pain_streak = 0

    if 'landmarker' not in st.session_state:
        st.session_state.landmarker = FaceLandmarker.create_from_options(options)
    if 'last_logged_status' not in st.session_state:
        st.session_state.last_logged_status = ""

    landmarker = st.session_state.landmarker

    # Pull the latest frame (already flipped + RGB) from the capture thread.
    frame_rgb, _frame_id, engine_error, fail_count = engine.latest()

    if engine_error:
        if engine_error == "no_camera":
            status_box.error("📷 No camera detected — connect a webcam and restart the app.")
        elif engine_error == "video_open_fail":
            status_box.error(f"📹 Could not open video file: {video_path}")
        elif engine_error == "video_read_fail":
            status_box.error("📹 Could not read frames from the video file.")
        else:  # not_delivering
            status_box.error("📷 Camera is open but not delivering frames — it's likely in use by another app or browser tab. Close other tabs/apps, then restart the app.")
        return

    if frame_rgb is None:
        status_box.warning(f"📷 Waiting for camera frame... ({fail_count}/10)")
        return

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    latest_status = "Patient Profile Status: Stable & Calm"
    latest_color = "success"
    latest_score = 0
    face_lms = None

    if (result.face_landmarks and result.face_blendshapes
            and len(result.face_landmarks) > 0 and len(result.face_blendshapes) > 0):
        face_lms = result.face_landmarks[0]
        # Extract blendshape scores for the first (most prominent) face.
        # MediaPipe returns face_blendshapes[0] as a list of Category (0.10.x and 1.0.x),
        # but some builds wrap it in a Classifications object with .categories.
        blend_first = result.face_blendshapes[0]
        blend_cats = blend_first.categories if hasattr(blend_first, "categories") else blend_first
        scores = {c.category_name: c.score for c in blend_cats}
        raw_brow = blendshape_pair(scores, BLENDSHAPE_BROW)
        raw_squint = blendshape_pair(scores, BLENDSHAPE_SQUINT)
        raw_lip = blendshape_pair(scores, BLENDSHAPE_LIP)
        raw_smile = blendshape_pair(scores, BLENDSHAPE_SMILE)

        # --- CALIBRATION: average a short neutral window into the resting baseline ---
        if not st.session_state.calibrated or st.session_state.base_brow is None:
            st.session_state.calib_buffer.append((raw_brow, raw_squint, raw_lip, raw_smile))
            if len(st.session_state.calib_buffer) >= CALIB_FRAMES:
                buf = st.session_state.calib_buffer
                st.session_state.base_brow = _mean([x[0] for x in buf])
                st.session_state.base_eye = _mean([x[1] for x in buf])
                st.session_state.base_lip = _mean([x[2] for x in buf])
                st.session_state.base_smile = _mean([x[3] for x in buf])
                st.session_state.sigma_brow = _std([x[0] for x in buf])
                st.session_state.sigma_eye = _std([x[1] for x in buf])
                st.session_state.sigma_lip = _std([x[2] for x in buf])
                # Auto-tune the pain gates from this patient's resting variability.
                st.session_state.auto_thresh_brow = auto_gate(st.session_state.sigma_brow)
                st.session_state.auto_thresh_eye = auto_gate(st.session_state.sigma_eye)
                st.session_state.auto_thresh_lip = auto_gate(st.session_state.sigma_lip)
                st.session_state.calib_buffer = []
                st.session_state.calibrated = True
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                ab = st.session_state.auto_thresh_brow
                ae = st.session_state.auto_thresh_eye
                al = st.session_state.auto_thresh_lip
                st.session_state.event_logs.append(
                    f'<div style="color: #3b82f6; font-weight: bold; margin-bottom: 4px;">🔒 [{timestamp}] Neutral baseline calibrated from {CALIB_FRAMES} frames — auto-tuned pain gates: brow {ab}%, eye {ae}%, lip {al}%.</div>')
            latest_status = "🔒 Calibrating baseline — hold a neutral expression..."
            latest_color = "info"
        else:
            # --- SIGNAL: baseline-adjusted activation, expressed as percent ---
            brow_pct = max(0.0, (raw_brow - st.session_state.base_brow) * 100)
            eye_pct = max(0.0, (raw_squint - st.session_state.base_eye) * 100)
            lip_pct = max(0.0, (raw_lip - st.session_state.base_lip) * 100)
            smile_pct = max(0.0, (raw_smile - st.session_state.base_smile) * 100)

            # --- TEMPORAL SMOOTHING: exponential moving average over frames ---
            st.session_state.smoothed_brow += SMOOTH_ALPHA * (brow_pct - st.session_state.smoothed_brow)
            st.session_state.smoothed_eye += SMOOTH_ALPHA * (eye_pct - st.session_state.smoothed_eye)
            st.session_state.smoothed_lip += SMOOTH_ALPHA * (lip_pct - st.session_state.smoothed_lip)
            st.session_state.smoothed_smile += SMOOTH_ALPHA * (smile_pct - st.session_state.smoothed_smile)

            sm_brow = st.session_state.smoothed_brow
            sm_eye = st.session_state.smoothed_eye
            sm_lip = st.session_state.smoothed_lip
            sm_smile = st.session_state.smoothed_smile

            # --- HYSTERESIS CLASSIFIER with debounce ---
            # Pain latches on ANY 2 of the 3 pain signals (brow, eye, lip), so a single
            # weak signal (e.g. a high resting squint) can't block the alert.
            pain_hits = sum([
                sm_brow >= eff_brow,
                sm_eye >= eff_eye,
                sm_lip >= eff_lip,
            ])
            all_on = pain_hits >= 2 and sm_smile < PAIN_SMILE_GUARD
            pain_still = sum([
                sm_brow >= eff_brow * HYSTERESIS_MARGIN,
                sm_eye >= eff_eye * HYSTERESIS_MARGIN,
                sm_lip >= eff_lip * HYSTERESIS_MARGIN,
            ])
            any_off = pain_still < 2

            if st.session_state.pain_active:
                if any_off:
                    st.session_state.pain_streak += 1
                    if st.session_state.pain_streak >= PAIN_EXIT_STREAK:
                        st.session_state.pain_active = False
                        st.session_state.pain_streak = 0
                else:
                    st.session_state.pain_streak = 0
            else:
                if all_on:
                    st.session_state.pain_streak += 1
                    if st.session_state.pain_streak >= PAIN_ENTER_STREAK:
                        st.session_state.pain_active = True
                        st.session_state.pain_streak = 0
                else:
                    st.session_state.pain_streak = 0

            # --- STATUS EMISSION ---
            if st.session_state.pain_active:
                latest_score = int((sm_brow + sm_eye + sm_lip) / 3)
                latest_status = f"🚨 CRITICAL: ACUTE PHYSICAL PAIN DETECTED ({latest_score}%)"
                latest_color = "danger"
            elif sm_smile >= thresh_smile and sm_smile >= sm_lip:
                latest_status = "😊 Emotion Filter: Smile / Content"
                latest_color = "info"
                latest_score = int(sm_smile)
            elif sm_brow >= eff_brow:
                latest_status = "Emotion Filter: Frustration or Concentration"
                latest_color = "warning"
                latest_score = int(sm_brow)
            elif sm_lip >= eff_lip:
                latest_status = "Emotion Filter: Upper Lip Raise (sneer/disgust)"
                latest_color = "warning"
                latest_score = int(sm_lip)
            elif sm_eye >= eff_eye:
                latest_status = "Emotion Filter: Eye Squint (discomfort or tiredness)"
                latest_color = "warning"
                latest_score = int(sm_eye)
            else:
                latest_status = "Patient Profile Status: Stable & Calm"
                latest_color = "success"
                latest_score = 0
    else:
        # No face: never report "stable" on missing data; release any latched state.
        st.session_state.pain_active = False
        st.session_state.pain_streak = 0
        st.session_state.smoothed_brow = st.session_state.smoothed_eye = st.session_state.smoothed_lip = st.session_state.smoothed_smile = 0.0
        latest_status = "⚠️ No face detected — confirm patient is in view"
        latest_color = "info"
        latest_score = 0

    # --- LIVE SIGNAL READOUT & CONFIDENCE METER ---
    sm_brow = st.session_state.smoothed_brow
    sm_eye = st.session_state.smoothed_eye
    sm_lip = st.session_state.smoothed_lip
    sm_smile = st.session_state.smoothed_smile
    confidence = int(min(max((sm_brow + sm_eye + sm_lip) / 3, 0), 100))

    # Publish the pain-region heatmap so the capture thread can draw it each frame.
    # The overlay is opt-in: off by default for the clean caregiver view.
    if face_lms is not None and show_heatmap:
        engine.set_overlay([
            (region_points(face_lms, HEAT_BROW), heat_color(sm_brow)),
            (region_points(face_lms, HEAT_EYE), heat_color(sm_eye)),
            (region_points(face_lms, HEAT_LIP), heat_color(sm_lip)),
        ])
    else:
        engine.set_overlay(None)

    with chart_box:
        st.progress(min(max(sm_brow, 0), 100) / 100.0, text=f"Brow contraction (AU4): {int(sm_brow)}% (gate {eff_brow:g}%)")
        st.progress(min(max(sm_eye, 0), 100) / 100.0, text=f"Eye squint (AU6/7): {int(sm_eye)}% (gate {eff_eye:g}%)")
        st.progress(min(max(sm_lip, 0), 100) / 100.0, text=f"Upper lip raise (AU10): {int(sm_lip)}% (gate {eff_lip:g}%)")
        st.progress(min(max(sm_smile, 0), 100) / 100.0, text=f"Smile (AU12): {int(sm_smile)}% (gate {thresh_smile}%)")
        st.metric("Pain confidence", f"{confidence}%")

    # Confidence history for the trend chart (capped to avoid unbounded growth).
    st.session_state.history.append(confidence)
    if len(st.session_state.history) > HISTORY_MAX:
        st.session_state.history = st.session_state.history[-HISTORY_MAX:]
    with trend_box:
        st.caption("Pain confidence over time")
        chart_df = pd.DataFrame({
            "t": range(len(st.session_state.history)),
            "confidence": st.session_state.history,
        })
        st.vega_lite_chart(
            chart_df,
            {
                "mark": "line",
                "encoding": {
                    "x": {"field": "t", "type": "quantitative",
                          "axis": {"labels": False, "title": None, "ticks": False}},
                    "y": {"field": "confidence", "type": "quantitative",
                          "scale": {"domain": [0, 100]}, "axis": {"title": None}},
                },
                "height": 150,
            },
            width="stretch",
        )

    # --- DYNAMIC NOTIFICATION DISPATCH ENGINE ---
    if latest_color == "danger":
        if not st.session_state.alarm_active:
            # Rising edge: fire the on-device alarm once, and relay over the LAN if intranet mode.
            audio_box.audio(alarm_tone(), sample_rate=44100, autoplay=True)
            st.session_state.alarm_active = True
            if not network_mode and relay_url.strip():
                ok_relay, detail = relay_alert(latest_status, relay_url.strip())
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                if ok_relay:
                    st.session_state.event_logs.append(
                        f'<div style="color: #22d3ee; margin-bottom: 4px;">📡 [{stamp}] Alert {detail}.</div>')
                else:
                    st.session_state.event_logs.append(
                        f'<div style="color: #f59e0b; margin-bottom: 4px;">📡 [{stamp}] Alert {detail}.</div>')
    else:
        st.session_state.alarm_active = False
        # No audio rendered when calm — the audio container stays empty (no layout shift).

    # Fixed-height status banner — every message renders at the same height so the
    # telemetry card (and the log card below it) never shifts up/down.
    status_colors = {
        "danger": ("#450a0a", "#fecaca", "#ef4444"),
        "warning": ("#451a03", "#fed7aa", "#f97316"),
        "info": ("#172554", "#bfdbfe", "#3b82f6"),
        "success": ("#052e16", "#bbf7d0", "#22c55e"),
    }
    status_bg, status_fg, status_border = status_colors.get(latest_color, status_colors["info"])
    status_box.markdown(
        f'<div style="background:{status_bg}; color:{status_fg}; border:1px solid {status_border}; '
        f'border-radius:8px; padding:10px 14px; height:64px; margin-bottom:14px; '
        f'display:flex; align-items:center; font-weight:600; overflow:hidden;">{latest_status}</div>',
        unsafe_allow_html=True)

    # Update and stamp state transitions in the scrollable log window
    if latest_status != st.session_state.last_logged_status:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if latest_color == "danger":
            log_style = f'<div style="color: #ef4444; font-weight: bold; margin-bottom: 4px;">🚨 [{timestamp}] {latest_status}</div>'
        elif latest_color == "warning":
            log_style = f'<div style="color: #f97316; margin-bottom: 4px;">⚠️ [{timestamp}] {latest_status}</div>'
        elif latest_color == "info":
            log_style = f'<div style="color: #94a3b8; margin-bottom: 4px;">👁️ [{timestamp}] {latest_status}</div>'
        else:
            log_style = f'<div style="color: #10b981; margin-bottom: 4px;">🟢 [{timestamp}] {latest_status}</div>'

        st.session_state.event_logs.append(log_style)
        st.session_state.last_logged_status = latest_status

    # Render the audit log every frame (string is cached, so the box never flashes out).
    if st.session_state.log_render_count != len(st.session_state.event_logs):
        st.session_state.log_html = "".join(reversed(st.session_state.event_logs))
        st.session_state.log_render_count = len(st.session_state.event_logs)
    log_container.markdown(f'<div class="log-box">{st.session_state.log_html}</div>', unsafe_allow_html=True)


live_dashboard()
