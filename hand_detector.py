"""
MediaPipe Hand Landmarker → finger / hand yoga cues.

Draws the hand skeleton onto the crop and returns Movement values
(currently finger extension → cricket).
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from movements.detectors import detect_finger_extension
from movements.mapping import Movement

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "Models", "hand_landmarker.task"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandDetector:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def draw_hand(self, person_crop, hand_landmarks):
        height, width, _ = person_crop.shape
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            cv2.circle(person_crop, (x, y), 4, (0, 255, 0), -1)

        for start, end in CONNECTIONS:
            cv2.line(person_crop, points[start], points[end], (255, 0, 0), 2)

    def detect(self, person_id: int, person_crop) -> list[Movement]:
        """Return hand-based yoga movements found in `person_crop`."""
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        movements: list[Movement] = []
        saw_finger_extension = False

        for hand_landmarks in result.hand_landmarks:
            self.draw_hand(person_crop, hand_landmarks)
            if detect_finger_extension(hand_landmarks):
                saw_finger_extension = True

        if saw_finger_extension:
            movements.append(Movement.FINGER_EXTENSION)

        return movements

    def close(self):
        self.landmarker.close()
