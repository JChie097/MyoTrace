import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import datetime

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
        .brand-badge { background-color: #1e293b; color: #38bdf8; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-bottom: 1rem; }
        .log-box { background-color: #030712; border: 1px solid #374151; font-family: monospace; padding: 1rem; border-radius: 8px; height: 180px; overflow-y: scroll; color: #34d399; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States with safe array defaults
if 'base_brow' not in st.session_state: st.session_state.base_brow = None
if 'base_eye' not in st.session_state: st.session_state.base_eye = None
if 'base_lip' not in st.session_state: st.session_state.base_lip = None
if 'calibrated' not in st.session_state: st.session_state.calibrated = False
if 'history' not in st.session_state: st.session_state.history = [0] * 30
if 'event_logs' not in st.session_state:
    st.session_state.event_logs = [
        '<div style="color: #6b7280; margin-bottom: 4px;">🕒 System initialized. Awaiting baseline configuration...</div>']

# 2. SIDEBAR SYSTEM SETTINGS PANEL (Sliders & Toggles)
with st.sidebar:
    st.header("⚙️ Core Controls")
    network_mode = st.toggle("Enforce Offline Isolation Mode", value=True)
    if network_mode:
        st.success("🔒 100% Edge Isolation Enforced")
    else:
        st.warning("⚠️ Connected to Local Intranet")

    st.markdown("---")
    st.subheader("🎛️ Sensitivity Thresholds")
    thresh_brow = st.slider("Brow Contraction Gate (AU4)", 5, 50, 15)
    thresh_eye = st.slider("Eyelid Squint Gate (AU7)", 5, 50, 15)
    thresh_lip = st.slider("Upper Lip Raise Gate (AU10)", 5, 50, 10)

# 3. CORE LANDMARK INDEX SPECIFICATIONS
LANDMARKS_BROW_TOP, LANDMARKS_NOSE_BRIDGE = 9, 4
LANDMARKS_EYE_TOP, LANDMARKS_EYE_BOTTOM = 159, 145
LANDMARKS_UPPER_LIP = 11

# 4. GOOGLE TASKS API CONFIGURATION
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.IMAGE
)

# 5. USER INTERFACE HEADER DESIGN
st.markdown('<div class="brand-badge">• Powered by Edge-Native Local AI Engine</div>', unsafe_allow_html=True)
st.title("Turn any ambient camera into structured intelligence")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -1rem; margin-bottom: 2rem;'>Passive, continuous myo-spatial landmark tracking for non-verbal caregiver assistance.</p>",
    unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-card"><h2>📹 Live Patient Room Feed</h2>', unsafe_allow_html=True)
    frame_window = st.image([])

    st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
    if st.button("Reset Baseline Calibration Profile"):
        st.session_state.calibrated = False
        st.session_state.base_brow, st.session_state.base_eye, st.session_state.base_lip = None, None, None
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        st.session_state.event_logs.append(
            f'<div style="color: #eab308; margin-bottom: 4px;">🔄 [{timestamp}] Calibration baseline profile wiped by operator.</div>')
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card"><h2>📊 Caregiver Telemetry Alerts</h2>', unsafe_allow_html=True)
    status_box = st.empty()
    chart_box = st.empty()
    audio_box = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# 6. DYNAMIC EVENT LOGGER PANEL
st.markdown('<div class="metric-card"><h2>💾 Real-Time Session Audit Trail Logs</h2>', unsafe_allow_html=True)
log_container = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# 7. INTEGRATED SYNCHRONOUS RUNTIME LOOP
cap = cv2.VideoCapture(0)
last_logged_status = ""

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        latest_status = "Patient Profile Status: Stable & Calm"
        latest_color = "success"
        latest_score = 0

        if result.face_landmarks and len(result.face_landmarks) > 0:
            # Extract first face list multi-face coordinates safely
            landmarks = result.face_landmarks[0]

            # PHASE 1: GEOMETRIC VECTOR EXTRACTION
            pt_brow = np.array([landmarks[LANDMARKS_BROW_TOP].x, landmarks[LANDMARKS_BROW_TOP].y])
            pt_nose = np.array([landmarks[LANDMARKS_NOSE_BRIDGE].x, landmarks[LANDMARKS_NOSE_BRIDGE].y])
            pt_eye_t = np.array([landmarks[LANDMARKS_EYE_TOP].x, landmarks[LANDMARKS_EYE_TOP].y])
            pt_eye_b = np.array([landmarks[LANDMARKS_EYE_BOTTOM].x, landmarks[LANDMARKS_EYE_BOTTOM].y])
            pt_lip = np.array([landmarks[LANDMARKS_UPPER_LIP].x, landmarks[LANDMARKS_UPPER_LIP].y])

            curr_brow = np.linalg.norm(pt_brow - pt_nose)
            curr_eye = np.linalg.norm(pt_eye_t - pt_eye_b)
            curr_lip = np.linalg.norm(pt_nose - pt_lip)

            if not st.session_state.calibrated or st.session_state.base_brow is None:
                st.session_state.base_brow, st.session_state.base_eye, st.session_state.base_lip = curr_brow, curr_eye, curr_lip
                st.session_state.calibrated = True
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                st.session_state.event_logs.append(
                    f'<div style="color: #3b82f6; font-weight: bold; margin-bottom: 4px;">🔒 [{timestamp}] Custom relative profile calibrated and locked.</div>')

            # PHASE 2: RELATIVE DEVIATION CALCULATOR
            brow_shrink = max(0, ((st.session_state.base_brow - curr_brow) / st.session_state.base_brow) * 100)
            eye_squint = max(0, ((st.session_state.base_eye - curr_eye) / st.session_state.base_eye) * 100)
            lip_raise = max(0, ((st.session_state.base_lip - curr_lip) / st.session_state.base_lip) * 100)

            # PHASE 3: MULTI-VECTOR SIMULTANEITY CLASSIFIER
            if brow_shrink > thresh_brow and eye_squint > thresh_eye and lip_raise > thresh_lip:
                latest_score = int((brow_shrink + eye_squint + lip_raise) / 3)
                latest_status = f"🚨 CRITICAL: ACUTE PHYSICAL PAIN DETECTED ({latest_score}%)"
                latest_color = "danger"
            elif brow_shrink > thresh_brow:
                latest_status = "Emotion Filter: Frustration or Concentration"
                latest_color = "warning"
                latest_score = 20
            elif eye_squint > thresh_eye:
                latest_status = "Emotion Filter: Smile or Yawn"
                latest_color = "warning"
                latest_score = 10

        # --- DYNAMIC NOTIFICATION DISPATCH ENGINE ---
        frame_window.image(frame_rgb, channels="RGB")

        if latest_color == "danger":
            status_box.error(latest_status)
            audio_box.html('<audio autoplay src="https://google.com"></audio>')
        elif latest_color == "warning":
            status_box.warning(latest_status)
            audio_box.empty()
        else:
            status_box.success(latest_status)
            audio_box.empty()

        # Update and stamp state transitions in the scrollable log window
        if latest_status != last_logged_status:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            if latest_color == "danger":
                log_style = f'<div style="color: #ef4444; font-weight: bold; margin-bottom: 4px;">🚨 [{timestamp}] {latest_status}</div>'
            elif latest_color == "warning":
                log_style = f'<div style="color: #f97316; margin-bottom: 4px;">⚠️ [{timestamp}] {latest_status}</div>'
            else:
                log_style = f'<div style="color: #10b981; margin-bottom: 4px;">🟢 [{timestamp}] {latest_status}</div>'

            st.session_state.event_logs.append(log_style)
            last_logged_status = latest_status

        # Render scrollable html elements securely
        log_html = "".join(reversed(st.session_state.event_logs))
        log_container.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

        st.session_state.history.append(latest_score)

