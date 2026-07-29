import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# 1. PAGE SETUP & STYLING
st.set_page_config(page_title="MyoTrace Caregiver Portal", layout="wide")

# Persistent Session States to hold values across browser refreshes
if 'base_brow' not in st.session_state: st.session_state.base_brow = None
if 'base_eye' not in st.session_state: st.session_state.base_eye = None
if 'base_lip' not in st.session_state: st.session_state.base_lip = None
if 'calibrated' not in st.session_state: st.session_state.calibrated = False
if 'history' not in st.session_state: st.session_state.history = [0] * 30

# 2. CORE LANDMARK INDEX SPECIFICATIONS (FACS Standards)
LANDMARKS_BROW_TOP = 9
LANDMARKS_NOSE_BRIDGE = 4
LANDMARKS_EYE_TOP = 159
LANDMARKS_EYE_BOTTOM = 145
LANDMARKS_UPPER_LIP = 11

# 3. GOOGLE TASKS API CONFIGURATION (Using synchronous IMAGE mode)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.IMAGE
)

# 4. USER INTERFACE LAYOUT DISPLAY
st.title("🛡️ MyoTrace Remote Caregiver Interface")
st.write("Passive, edge-computed machine learning framework for non-verbal distress tracking.")

col1, col2 = st.columns(2)

with col1:
    st.header("📹 Live Patient Room Feed")
    frame_window = st.image([])

    if st.button("Reset Baseline Calibration Profile", type="secondary"):
        st.session_state.calibrated = False
        st.session_state.base_brow = None
        st.session_state.base_eye = None
        st.session_state.base_lip = None
        st.rerun()

with col2:
    st.header("📊 Caregiver Telemetry Alerts")
    status_box = st.empty()
    chart_box = st.empty()
    audio_box = st.empty()

# 5. INTEGRATED SYNCHRONOUS RUNTIME LOOP
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Reverse matrix mirror effect for comfortable natural alignment
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert single frame synchronously for standard detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        # Initialize safe default visual states
        latest_status = "Patient Profile Status: Stable & Calm"
        latest_color = "success"
        latest_score = 0

        # Check if coordinates exist
        if result.face_landmarks and len(result.face_landmarks) > 0:
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

            # DYNAMIC CALIBRATION ENGINE
            if not st.session_state.calibrated or st.session_state.base_brow is None:
                st.session_state.base_brow = curr_brow
                st.session_state.base_eye = curr_eye
                st.session_state.base_lip = curr_lip
                st.session_state.calibrated = True

            # PHASE 2: RELATIVE DEVIATION CALCULATOR
            brow_shrink = max(0, ((st.session_state.base_brow - curr_brow) / st.session_state.base_brow) * 100)
            eye_squint = max(0, ((st.session_state.base_eye - curr_eye) / st.session_state.base_eye) * 100)
            lip_raise = max(0, ((st.session_state.base_lip - curr_lip) / st.session_state.base_lip) * 100)

            # PHASE 3: MULTI-VECTOR SIMULTANEITY CLASSIFIER
            if brow_shrink > 15 and eye_squint > 20 and lip_raise > 12:
                latest_score = int((brow_shrink + eye_squint + lip_raise) / 3)
                latest_status = f"🚨 CRITICAL: ACUTE PHYSICAL PAIN DETECTED ({latest_score}%)"
                latest_color = "danger"
            elif brow_shrink > 20:
                latest_status = "Emotion Filter: Frustration or Concentration"
                latest_color = "warning"
                latest_score = 25
            elif eye_squint > 25:
                latest_status = "Emotion Filter: Smile or Yawn"
                latest_color = "warning"
                latest_score = 15

        # --- DYNAMIC INTERFACE RENDER CONTAINER INJECTIONS ---
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

        # Append data point and slide graph history pipeline arrays
        st.session_state.history.append(latest_score)
        st.session_state.history = st.session_state.history[-30:]
        chart_box.line_chart(st.session_state.history)

cap.release()

