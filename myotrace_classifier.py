import cv2
import mediapipe as mp

# Headless / edge variant of the MyoTrace pain classifier.
# Same algorithm as app.py (the Streamlit caregiver portal): MediaPipe FaceBlendshape
# coefficients -> per-patient neutral baseline -> EMA smoothing -> hysteresis classifier.

# --- FACS-to-blendshape signal mapping (keep in sync with app.py) ---
BLENDSHAPE_BROW = ('browDownLeft', 'browDownRight')
BLENDSHAPE_SQUINT = ('eyeSquintLeft', 'eyeSquintRight')
BLENDSHAPE_LIP = ('mouthUpperUpLeft', 'mouthUpperUpRight')
BLENDSHAPE_SMILE = ('mouthSmileLeft', 'mouthSmileRight')

SMOOTH_ALPHA = 0.35
CALIB_FRAMES = 15
PAIN_ENTER_STREAK = 3
PAIN_EXIT_STREAK = 3
HYSTERESIS_MARGIN = 0.5

THRESH_BROW = 12
THRESH_EYE = 10
THRESH_LIP = 8
THRESH_SMILE = 20

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def blendshape_pair(scores, names):
    """Average a left/right blendshape pair into one 0..1 activation value."""
    return (scores.get(names[0], 0.0) + scores.get(names[1], 0.0)) / 2.0


class PainMonitor:
    """Per-patient calibrated pain detector over blendshape features."""

    def __init__(self):
        self.base_brow = self.base_eye = self.base_lip = self.base_smile = None
        self.calibrated = False
        self.calib_buffer = []
        self.smoothed_brow = self.smoothed_eye = self.smoothed_lip = self.smoothed_smile = 0.0
        self.pain_active = False
        self.pain_streak = 0

    def reset(self):
        self.__init__()

    def on_no_face(self):
        self.pain_active = False
        self.pain_streak = 0
        self.smoothed_brow = self.smoothed_eye = self.smoothed_lip = self.smoothed_smile = 0.0

    def update(self, scores):
        """Feed one frame's blendshape scores; returns (status, color, score, signals)."""
        raw_brow = blendshape_pair(scores, BLENDSHAPE_BROW)
        raw_squint = blendshape_pair(scores, BLENDSHAPE_SQUINT)
        raw_lip = blendshape_pair(scores, BLENDSHAPE_LIP)
        raw_smile = blendshape_pair(scores, BLENDSHAPE_SMILE)

        if not self.calibrated or self.base_brow is None:
            self.calib_buffer.append((raw_brow, raw_squint, raw_lip, raw_smile))
            if len(self.calib_buffer) >= CALIB_FRAMES:
                buf = self.calib_buffer
                self.base_brow = sum(x[0] for x in buf) / len(buf)
                self.base_eye = sum(x[1] for x in buf) / len(buf)
                self.base_lip = sum(x[2] for x in buf) / len(buf)
                self.base_smile = sum(x[3] for x in buf) / len(buf)
                self.calib_buffer = []
                self.calibrated = True
            return "Calibrating baseline - hold a neutral expression...", "info", 0, (0, 0, 0, 0)

        brow = max(0.0, (raw_brow - self.base_brow) * 100)
        eye = max(0.0, (raw_squint - self.base_eye) * 100)
        lip = max(0.0, (raw_lip - self.base_lip) * 100)
        smile = max(0.0, (raw_smile - self.base_smile) * 100)

        self.smoothed_brow += SMOOTH_ALPHA * (brow - self.smoothed_brow)
        self.smoothed_eye += SMOOTH_ALPHA * (eye - self.smoothed_eye)
        self.smoothed_lip += SMOOTH_ALPHA * (lip - self.smoothed_lip)
        self.smoothed_smile += SMOOTH_ALPHA * (smile - self.smoothed_smile)

        sm = (self.smoothed_brow, self.smoothed_eye, self.smoothed_lip, self.smoothed_smile)

        # Pain = ANY 2 of the 3 pain signals (brow, eye, lip), and explicitly NOT a smile.
        pain_hits = sum([sm[0] >= THRESH_BROW, sm[1] >= THRESH_EYE, sm[2] >= THRESH_LIP])
        all_on = pain_hits >= 2 and sm[3] < THRESH_SMILE
        pain_still = sum([
            sm[0] >= THRESH_BROW * HYSTERESIS_MARGIN,
            sm[1] >= THRESH_EYE * HYSTERESIS_MARGIN,
            sm[2] >= THRESH_LIP * HYSTERESIS_MARGIN,
        ])
        any_off = pain_still < 2

        if self.pain_active:
            if any_off:
                self.pain_streak += 1
                if self.pain_streak >= PAIN_EXIT_STREAK:
                    self.pain_active = False
                    self.pain_streak = 0
            else:
                self.pain_streak = 0
        else:
            if all_on:
                self.pain_streak += 1
                if self.pain_streak >= PAIN_ENTER_STREAK:
                    self.pain_active = True
                    self.pain_streak = 0
            else:
                self.pain_streak = 0

        if self.pain_active:
            score = int((sm[0] + sm[1] + sm[2]) / 3)
            return f"CRITICAL: ACUTE PHYSICAL PAIN DETECTED ({score}%)", "danger", score, sm
        if sm[3] >= THRESH_SMILE and sm[3] >= sm[2]:
            return "Emotion Filter: Smile / Content", "info", int(sm[3]), sm
        if sm[0] >= THRESH_BROW:
            return "Emotion Filter: Frustration or Concentration", "warning", int(sm[0]), sm
        if sm[2] >= THRESH_LIP:
            return "Emotion Filter: Upper Lip Raise (sneer/disgust)", "warning", int(sm[2]), sm
        if sm[1] >= THRESH_EYE:
            return "Emotion Filter: Eye Squint (discomfort or tiredness)", "warning", int(sm[1]), sm
        return "Patient Profile Status: Stable & Calm", "success", 0, sm


COLORS = {"danger": (0, 0, 255), "warning": (0, 165, 255), "info": (180, 180, 180), "success": (0, 255, 0)}

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
)

monitor = PainMonitor()
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    print("[SYSTEM] MyoTrace blendshape pain classifier active. 'r' = recalibrate, 'q' = quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        if (result.face_landmarks and result.face_blendshapes
                and len(result.face_landmarks) > 0 and len(result.face_blendshapes) > 0):
            blend_first = result.face_blendshapes[0]
            blend_cats = blend_first.categories if hasattr(blend_first, "categories") else blend_first
            scores = {c.category_name: c.score for c in blend_cats}
            status, color, score, sm = monitor.update(scores)
        else:
            monitor.on_no_face()
            status, color, score, sm = "No face detected", "info", 0, (0, 0, 0, 0)

        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[color], 2)
        cv2.putText(frame, f"brow {int(sm[0])}%  eye {int(sm[1])}%  lip {int(sm[2])}%  smile {int(sm[3])}%",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('MyoTrace Pain Classifier', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('r'):
            monitor.reset()
            print("[SYSTEM] Re-calibrating baseline...")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
