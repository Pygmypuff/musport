"""
Shared base for the MediaPipe landmark detectors.

Owns what pose_detector.py and hand_detector.py have in common: finding
(or downloading) the .task model, creating the landmarker, converting
an OpenCV crop into a MediaPipe image, and closing the landmarker.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models"
)


def ensure_model(filename: str, url: str) -> str:
    """Return the path to a model in Models/, downloading it if missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


class LandmarkDetector:
    """Base class: subclasses set the model and implement detect()."""

    MODEL_FILE: str
    MODEL_URL: str

    def __init__(self, landmarker_class, options_class, **options):
        model_path = ensure_model(self.MODEL_FILE, self.MODEL_URL)
        self.landmarker = landmarker_class.create_from_options(
            options_class(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                **options,
            )
        )

    def find_landmarks(self, person_crop):
        """Run the landmarker on a BGR crop and return its raw result."""
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()
