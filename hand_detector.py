"""
MediaPipe Hand Landmarker wrapper for per-person hand gesture recognition.

Takes a cropped person frame and returns the gestures being made, one per
hand, drawing the hand skeleton onto the crop as it goes.
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "Models", "hand_landmarker.task"
)

UNKNOWN = "Unknown"

# List of all connections of the hand
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class HandDetector:
    def __init__(self):
        # IMAGE mode: each person's crop is an independent image, so there is
        # no tracking state to carry between people.
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    # Checks if a finger is extended using the distance from the tip and base to the wrist.
    def is_finger_extended(self, landmarks, tip, pip):
        """
        Determine whether a finger is extended.

        tip = fingertip landmark
        pip = middle joint of the finger
        """

        wrist = landmarks[0]

        tip_distance = (
            (landmarks[tip].x - wrist.x) ** 2 +
            (landmarks[tip].y - wrist.y) ** 2
        )

        pip_distance = (
            (landmarks[pip].x - wrist.x) ** 2 +
            (landmarks[pip].y - wrist.y) ** 2
        )

        return tip_distance > pip_distance

    # Based on which fingers are extended, it will return a gesture
    def classify_gesture(self, index, middle, ring, pinky):
        fingers = (index, middle, ring, pinky)

        if fingers == (0, 0, 0, 0):
            return "Fist"
        if fingers == (1, 0, 0, 0):
            return "Index"
        if fingers == (0, 1, 0, 0):
            return "Middle"
        if fingers == (0, 0, 1, 0):
            return "Ring"
        if fingers == (0, 0, 0, 1):
            return "Pinky"
        if fingers == (1, 1, 0, 1):
            return "Forks up!"
        if fingers == (1, 1, 0, 0):
            return "Peace"
        if fingers == (1, 0, 0, 1):
            return "Spider-Man"

        return UNKNOWN

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

    def detect(self, person_id: int, person_crop):
        """Return a list of gestures, one per hand found in `person_crop`.

        `person_crop` is a BGR numpy array (frame[y1:y2, x1:x2]).
        """
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        # List of the gestures (if there are two hands, or one, or zero)
        gestures = []

        for hand_landmarks in result.hand_landmarks:
            index_extended = self.is_finger_extended(hand_landmarks, 8, 6)
            middle_extended = self.is_finger_extended(hand_landmarks, 12, 10)
            ring_extended = self.is_finger_extended(hand_landmarks, 16, 14)
            pinky_extended = self.is_finger_extended(hand_landmarks, 20, 18)

            gestures.append(
                self.classify_gesture(
                    index_extended, middle_extended, ring_extended, pinky_extended
                )
            )

            self.draw_hand(person_crop, hand_landmarks)

        return gestures

    def close(self):
        self.landmarker.close()
