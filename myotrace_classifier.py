import cv2
import mediapipe as mp
import numpy as np

# 1. CORE LANDMARK INDEX SPECIFICATIONS (FACS Standards)
LANDMARKS_BROW_TOP = 9  # AU4 marker (Brow center glabella)
LANDMARKS_EYE_TOP = 159  # AU6/7 upper eyelid center
LANDMARKS_EYE_BOTTOM = 145  # AU6/7 lower eyelid center
LANDMARKS_NOSE_BRIDGE = 4  # AU9/10 upper nose wrinkler hub
LANDMARKS_UPPER_LIP = 11  # AU10 upper lip lifter hub

# 2. GOOGLE TASKS API CONFIGURATION
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 3. SELF-CALIBRATING SYSTEM VARIABLES
base_brow_dist = None
base_eye_dist = None
base_nose_lip_dist = None

latest_status_text = "System Initializing... Look Neutral"
latest_status_color = (255, 255, 255)


def tracking_callback(result, output_image, timestamp_ms):
    """THE CORE ENGINE LOGIC: Automatically self-calibrates on frame 1 to eliminate bias."""
    global latest_status_text, latest_status_color, base_brow_dist, base_eye_dist, base_nose_lip_dist

    raw_list = []
    if hasattr(result, 'face_landmarks') and result.face_landmarks:
        raw_list = result.face_landmarks

    if raw_list is not None and len(raw_list) > 0:
        landmarks = raw_list[0]

        # --- PHASE 1: GEOMETRIC VECTOR EXTRACTION ---
        pt_brow = np.array([landmarks[LANDMARKS_BROW_TOP].x, landmarks[LANDMARKS_BROW_TOP].y])
        pt_nose = np.array([landmarks[LANDMARKS_NOSE_BRIDGE].x, landmarks[LANDMARKS_NOSE_BRIDGE].y])
        pt_eye_t = np.array([landmarks[LANDMARKS_EYE_TOP].x, landmarks[LANDMARKS_EYE_TOP].y])
        pt_eye_b = np.array([landmarks[LANDMARKS_EYE_BOTTOM].x, landmarks[LANDMARKS_EYE_BOTTOM].y])
        pt_lip = np.array([landmarks[LANDMARKS_UPPER_LIP].x, landmarks[LANDMARKS_UPPER_LIP].y])

        # Calculate live distances
        curr_brow_dist = np.linalg.norm(pt_brow - pt_nose)
        curr_eye_dist = np.linalg.norm(pt_eye_t - pt_eye_b)
        curr_nose_lip_dist = np.linalg.norm(pt_nose - pt_lip)

        # --- DYNAMIC INITIAL CALIBRATION INJECTOR ---
        # If no profile exists yet, save the current frame numbers as your custom resting face baseline
        if base_brow_dist is None:
            base_brow_dist = curr_brow_dist
            base_eye_dist = curr_eye_dist
            base_nose_lip_dist = curr_nose_lip_dist
            return

        # --- PHASE 2: RELATIVE DEVIATION CALCULATOR ---
        # Calculates changes relative to YOUR face structure, not generic averages
        brow_shrinkage = max(0, ((base_brow_dist - curr_brow_dist) / base_brow_dist) * 100)
        eye_squint = max(0, ((base_eye_dist - curr_eye_dist) / base_eye_dist) * 100)
        nose_wrinkle = max(0, ((base_nose_lip_dist - curr_nose_lip_dist) / base_nose_lip_dist) * 100)

        # --- PHASE 3: MULTI-VECTOR SIMULTANEITY CLASSIFIER ---
        if brow_shrinkage > 22 and eye_squint > 25 and nose_wrinkle > 15:
            pain_confidence = (brow_shrinkage + eye_squint + nose_wrinkle) / 3
            latest_status_text = f"CRITICAL: ACUTE PHYSICAL PAIN DETECTED ({pain_confidence:.1f}%)"
            latest_status_color = (0, 0, 255)  # Warning Red
        elif brow_shrinkage > 20:
            latest_status_text = "Emotion Filter: Frustration or Concentration"
            latest_status_color = (255, 128, 0)  # Neutral Orange
        elif eye_squint > 25:
            latest_status_text = "Emotion Filter: Smile or Yawn"
            latest_status_color = (255, 128, 0)  # Neutral Orange
        else:
            latest_status_text = "Patient Profile Status: Stable & Calm"
            latest_status_color = (0, 255, 0)  # Safe Green


# Configure configuration properties
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=tracking_callback
)

# 4. HARDWARE VIDEO STREAM PIPELINE
cap = cv2.VideoCapture(0)
frame_timestamp = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    print("\n[SYSTEM] MyoTrace Multi-Vector Pain Classifier Active.")
    print("Look at the camera window to check live video text overlays. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp += 1
        landmarker.detect_async(mp_image, frame_timestamp)

        # Draw the live status results directly on screen frame
        cv2.putText(frame, latest_status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, latest_status_color, 2)

        # Render local preview frames window
        cv2.imshow('MyoTrace Multi-Vector Backend Core', frame)

        # Press 'r' at any point while running to re-calibrate your baseline
        key = cv2.waitKey(5) & 0xFF
        if key == ord('r'):
            base_brow_dist = None
            print("\n[SYSTEM] Re-calibrating profile baseline...")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()