"""
MediaPipe Pose Landmarker → senior-living yoga movements.

Takes a cropped person frame and returns Movement values recognized
from body pose (side bend, bend down, etc.). Hand-only moves live in
hand_detector.py.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from movements.detectors import POSE_DETECTORS
from movements.mapping import Movement, display_name

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "Models", "pose_landmarker_full.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmark model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class PoseDetector:
    def __init__(self):
        ensure_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        # Previous landmarks per person for temporal cues (neck turn, heel raise).
        self._prev_landmarks: dict[int, object] = {}

    def detect(self, person_id: int, person_crop) -> list[Movement]:
        """Return yoga movements held in `person_crop`."""
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            self._prev_landmarks.pop(person_id, None)
            return []

        landmarks = result.pose_landmarks[0]
        prev = self._prev_landmarks.get(person_id)

        found: list[Movement] = []
        for movement, detector in POSE_DETECTORS:
            if detector(landmarks, prev):
                found.append(movement)

        # Side bend tilts the shoulders/head — don't also fire twist/neck.
        if Movement.SIDE_BEND in found:
            found = [
                m
                for m in found
                if m not in (Movement.NECK_TURN, Movement.TORSO_TURN)
            ]
        # Deep forward fold often looks like a short torso; prefer rain over water.
        if Movement.BEND_DOWN in found and Movement.TORSO_TURN in found:
            found = [m for m in found if m != Movement.TORSO_TURN]

        self._prev_landmarks[person_id] = landmarks
        return found

    def close(self):
        self.landmarker.close()
        self._prev_landmarks.clear()


__all__ = ["PoseDetector", "Movement", "display_name"]
