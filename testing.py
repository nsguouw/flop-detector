import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

print(f"opencv:    {cv2.__version__}")
print(f"numpy:     {np.__version__}")
print(f"mediapipe: {mp.__version__}")

# Check model file exists
if Path("pose_landmarker.task").exists():
    print("pose_landmarker.task: found")
else:
    print("pose_landmarker.task: MISSING")

# Try loading the landmarker
try:
    base_options = mp_python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    print("mediapipe landmarker: loaded successfully")
    landmarker.close()
except Exception as e:
    print(f"mediapipe landmarker: FAILED — {e}")