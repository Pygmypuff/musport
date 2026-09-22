"""
MediaPipe Pose Landmarker wrapper for per-person pose recognition.

Takes a cropped person frame from YOLO and returns the name of the pose
being held, or None if no recognized pose is detected.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

BOTH_HANDS_UP = "both_hands_up"

# A landmark below this presence score is treated as not reliably seen.
MIN_LANDMARK_PRESENCE = 0.5


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def is_both_hands_up(landmarks) -> bool:
    """Both wrists raised above the head (y grows downward).

    The nose is the reference rather than the shoulders: arms held
    straight out sideways put the wrists a hair above the shoulders,
    which would otherwise read as hands-up.
    """
    required = [
        landmarks[PoseLandmark.NOSE],
        landmarks[PoseLandmark.LEFT_WRIST],
        landmarks[PoseLandmark.RIGHT_WRIST],
    ]
    if any(lm.presence < MIN_LANDMARK_PRESENCE for lm in required):
        return False

    nose, left_wrist, right_wrist = required
    return left_wrist.y < nose.y and right_wrist.y < nose.y


class PoseDetector:
    def __init__(self):
        ensure_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, person_id: int, person_crop):
        """Return the name of the pose in `person_crop`, or None.

        `person_crop` is a BGR numpy array (frame[y1:y2, x1:x2]).
        """
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        if is_both_hands_up(landmarks):
            return BOTH_HANDS_UP

        return None

    def close(self):
        self.landmarker.close()
