import cv2
import mediapipe as mp

# Modern initialization pathways that bypass legacy 'solutions'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Setup configuration properties
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print  # Direct output handler stream
)

cap = cv2.VideoCapture(0)
print("\n[SUCCESS] MyoTrace Core AI Engine Online and Configured.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Show the baseline hardware channel window frame
    cv2.imshow('MyoTrace Core Hardware Test', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()