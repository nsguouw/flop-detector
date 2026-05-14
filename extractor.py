"""
extractor.py
------------
Runs MediaPipe Pose Landmarker on a video clip and extracts flop-relevant
features frame by frame. Uses the new mediapipe.tasks API (0.10+).
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[warning] mediapipe not installed — run: pip install mediapipe")

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose, 33 keypoints)
# ---------------------------------------------------------------------------
LM = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

DEFAULT_MODEL = "pose_landmarker.task"


def _lm_xy(landmarks, key, w, h):
    lm = landmarks[LM[key]]
    return lm.x * w, lm.y * h


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _velocity(prev, curr):
    if prev is None or curr is None:
        return 0.0
    return _dist(prev, curr)


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------

def extract_frame_features(landmarks, prev_landmarks, w, h):
    nose        = _lm_xy(landmarks, "nose", w, h)
    l_shoulder  = _lm_xy(landmarks, "left_shoulder", w, h)
    r_shoulder  = _lm_xy(landmarks, "right_shoulder", w, h)
    l_hip       = _lm_xy(landmarks, "left_hip", w, h)
    r_hip       = _lm_xy(landmarks, "right_hip", w, h)
    l_ankle     = _lm_xy(landmarks, "left_ankle", w, h)
    r_ankle     = _lm_xy(landmarks, "right_ankle", w, h)

    shoulder_mid = _midpoint(l_shoulder, r_shoulder)
    hip_mid      = _midpoint(l_hip, r_hip)
    ankle_mid    = _midpoint(l_ankle, r_ankle)

    cog = (
        shoulder_mid[0] * 0.3 + hip_mid[0] * 0.5 + ankle_mid[0] * 0.2,
        shoulder_mid[1] * 0.3 + hip_mid[1] * 0.5 + ankle_mid[1] * 0.2,
    )

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    trunk_angle = math.degrees(math.atan2(dx, -dy))

    features = {
        "nose_y":      nose[1] / h,
        "cog_y":       cog[1] / h,
        "trunk_angle": trunk_angle,
    }

    if prev_landmarks:
        p_nose     = _lm_xy(prev_landmarks, "nose", w, h)
        p_shoulder = _midpoint(
            _lm_xy(prev_landmarks, "left_shoulder", w, h),
            _lm_xy(prev_landmarks, "right_shoulder", w, h),
        )
        p_hip = _midpoint(
            _lm_xy(prev_landmarks, "left_hip", w, h),
            _lm_xy(prev_landmarks, "right_hip", w, h),
        )

        nose_vel     = _velocity(p_nose, nose) / h
        shoulder_vel = _velocity(p_shoulder, shoulder_mid) / h
        hip_vel      = _velocity(p_hip, hip_mid) / h
        head_snap    = max(0.0, nose_vel - hip_vel)

        p_cog_y   = (p_shoulder[1] * 0.3 + p_hip[1] * 0.5 + ankle_mid[1] * 0.2) / h
        fall_rate = cog[1] / h - p_cog_y

        p_dx = p_shoulder[0] - p_hip[0]
        p_dy = p_shoulder[1] - p_hip[1]
        p_trunk     = math.degrees(math.atan2(p_dx, -p_dy))
        trunk_delta = trunk_angle - p_trunk

        features.update({
            "nose_velocity":     nose_vel,
            "shoulder_velocity": shoulder_vel,
            "hip_velocity":      hip_vel,
            "head_snap":         head_snap,
            "fall_rate":         fall_rate,
            "trunk_delta":       trunk_delta,
        })
    else:
        features.update({
            "nose_velocity":     0.0,
            "shoulder_velocity": 0.0,
            "hip_velocity":      0.0,
            "head_snap":         0.0,
            "fall_rate":         0.0,
            "trunk_delta":       0.0,
        })

    return features


# ---------------------------------------------------------------------------
# Clip-level aggregation
# ---------------------------------------------------------------------------

def aggregate_features(frame_features):
    if not frame_features:
        return {}

    keys = [k for k in frame_features[0] if k not in ("nose_y", "cog_y", "frame_idx")]
    agg = {}
    for k in keys:
        vals = [f[k] for f in frame_features]
        agg[f"{k}_max"]  = float(np.max(vals))
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"]  = float(np.std(vals))

    agg["peak_fall_rate"] = float(np.max([f["fall_rate"] for f in frame_features]))
    agg["min_nose_y"]     = float(np.min([f["nose_y"] for f in frame_features]))
    return agg


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_clip(video_path: str, model_path: str = DEFAULT_MODEL, max_frames: int = 120) -> dict:
    video_path = str(video_path)
    result = {
        "clip_path":      video_path,
        "fps":            None,
        "frame_count":    0,
        "duration_s":     None,
        "frame_features": [],
        "clip_features":  {},
        "label":          None,
        "notes":          "",
    }

    if not MP_AVAILABLE:
        print("[extractor] mediapipe unavailable")
        return result

    if not Path(model_path).exists():
        print(f"[extractor] model file not found: {model_path}")
        print("  Download it with:")
        print("  curl -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[extractor] could not open {video_path}")
        return result

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result["fps"]         = fps
    result["frame_count"] = min(total, max_frames)
    result["duration_s"]  = result["frame_count"] / fps

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_features = []
    prev_landmarks = None
    frame_ms       = int(1000 / fps)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(min(total, max_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = i * frame_ms

            detection = landmarker.detect_for_video(mp_image, timestamp)

            if detection.pose_landmarks:
                lms   = detection.pose_landmarks[0]
                feats = extract_frame_features(lms, prev_landmarks, w, h)
                feats["frame_idx"] = i
                frame_features.append(feats)
                prev_landmarks = lms
            else:
                prev_landmarks = None

    cap.release()

    result["frame_features"] = frame_features
    result["clip_features"]  = aggregate_features(frame_features)
    return result


# ---------------------------------------------------------------------------
# Heuristic flop score
# ---------------------------------------------------------------------------

def heuristic_flop_score(clip_features: dict) -> float:
    if not clip_features:
        return 0.0

    score = 0.0

    if clip_features.get("head_snap_max", 0) > 0.05:
        score += 0.3

    peak_fall = clip_features.get("peak_fall_rate", 0)
    if peak_fall > 0.04:
        score += 0.25
    if peak_fall > 0.07:
        score += 0.15

    if abs(clip_features.get("trunk_delta_max", 0)) > 15:
        score += 0.2

    if clip_features.get("min_nose_y", 1.0) > 0.7:
        score += 0.1

    return min(score, 1.0)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python extractor.py <video_path>")
        sys.exit(1)

    result = process_clip(path)
    score  = heuristic_flop_score(result["clip_features"])
    print(f"\nClip:     {path}")
    print(f"Frames:   {result['frame_count']} @ {result['fps']:.1f}fps")
    print(f"Features detected in {len(result['frame_features'])} frames")
    print(f"\nClip features:")
    print(json.dumps(result["clip_features"], indent=2))
    print(f"\nHeuristic flop score: {score:.2f} / 1.00  {'HIGH' if score > 0.5 else ''}")
