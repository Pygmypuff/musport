"""
MediaPipe Pose Landmarker → senior-living yoga movements.

Takes a cropped person frame and returns Movement values recognized
from body pose (side bend, bend down, etc.). Hand-only moves live in
hand_detector.py.
"""

from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from landmark_detector import LandmarkDetector
from movements.detectors import POSE_DETECTORS
from movements.mapping import Movement


class PoseDetector(LandmarkDetector):
    MODEL_FILE = "pose_landmarker_full.task"
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )

    def __init__(self):
        super().__init__(PoseLandmarker, PoseLandmarkerOptions, num_poses=1)
        # Previous landmarks per person for temporal cues (neck turn, heel raise).
        self._prev_landmarks: dict[int, object] = {}

    def detect(self, person_id: int, person_crop) -> list[Movement]:
        """Return yoga movements held in `person_crop`."""
        result = self.find_landmarks(person_crop)
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
        super().close()
        self._prev_landmarks.clear()
