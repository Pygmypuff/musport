"""
MediaPipe Hand Landmarker → finger / hand yoga cues.

Draws the hand skeleton onto the crop and returns Movement values
(currently finger extension → cricket).
"""

import cv2
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from detection.landmark_detector import LandmarkDetector
from movements.detectors import detect_finger_extension
from movements.mapping import Movement

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandDetector(LandmarkDetector):
    MODEL_FILE = "hand_landmarker.task"
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

    def __init__(self):
        super().__init__(HandLandmarker, HandLandmarkerOptions, num_hands=2)

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
        hands = self.find_landmarks(person_crop).hand_landmarks

        for hand_landmarks in hands:
            self.draw_hand(person_crop, hand_landmarks)

        if any(detect_finger_extension(hand) for hand in hands):
            return [Movement.FINGER_EXTENSION]
        return []
