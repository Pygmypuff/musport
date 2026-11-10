"""
MediaPipe Pose Landmarker → senior-living yoga movements.

Takes a cropped person frame and returns Movement values recognized
from body pose (side bend, bend down, etc.). Hand-only moves live in
hand_detector.py.
"""

from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from detection.facing import PersonFacing
from detection.landmark_detector import LandmarkDetector
from movements.detectors import (
    POSE_DETECTORS,
    leg_extension_intensity,
    shoulder_yaw_deg,
    torso_turn_score,
)
from movements.mapping import Movement


class PoseDetector(LandmarkDetector):
    # Every movement detect() can return.
    MOVEMENTS = tuple(movement for movement, _ in POSE_DETECTORS)

    MODEL_FILE = "pose_landmarker_full.task"
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )

    def __init__(self):
        super().__init__(PoseLandmarker, PoseLandmarkerOptions, num_poses=1)
        # Previous landmarks per person for temporal cues (neck turn, heel raise).
        self._prev_landmarks: dict[int, object] = {}
        self._leg_extension_intensities: dict[int, float] = {}
        self._torso_turn_intensities: dict[int, float] = {}
        # Which way each person sits; kept while they're out of view.
        self._facing: dict[int, PersonFacing] = {}

    def facing(self, person_id: int) -> PersonFacing:
        if person_id not in self._facing:
            self._facing[person_id] = PersonFacing()
        return self._facing[person_id]

    def detect(self, person_id: int, person_crop, now: float) -> list[Movement]:
        """Return yoga movements held in `person_crop`.

        Also updates the person's facing (see facing()), which the
        gesture pipeline checks afterwards.
        """
        result = self.find_landmarks(person_crop)
        if not result.pose_landmarks:
            self._prev_landmarks.pop(person_id, None)
            self._leg_extension_intensities.pop(person_id, None)
            self._torso_turn_intensities.pop(person_id, None)
            return []

        landmarks = result.pose_landmarks[0]
        # 3D landmarks in metres; only shoulder direction uses them so far.
        world = (
            result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        )
        yaw = shoulder_yaw_deg(landmarks, world)
        facing = self.facing(person_id)
        if yaw is not None:
            facing.add(yaw, now)
        resting_yaw = facing.resting_yaw

        self._leg_extension_intensities[person_id] = leg_extension_intensity(landmarks)
        self._torso_turn_intensities[person_id] = torso_turn_score(yaw, resting_yaw)
        prev = self._prev_landmarks.get(person_id)

        found: list[Movement] = []
        for movement, detector in POSE_DETECTORS:
            if movement is Movement.TORSO_TURN:
                detected = detector(
                    landmarks, prev, world=world, resting_yaw=resting_yaw
                )
            else:
                detected = detector(landmarks, prev)
            if detected:
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

    def leg_extension_intensity(self, person_id: int) -> float:
        """Current 0–1 Wind volume scale for a tracked person."""
        return self._leg_extension_intensities.get(person_id, 0.0)

    def torso_turn_intensity(self, person_id: int) -> float:
        """Current 0–1 Water volume scale for a tracked person."""
        return self._torso_turn_intensities.get(person_id, 0.0)

    def forget(self, person_id: int):
        self._prev_landmarks.pop(person_id, None)
        self._leg_extension_intensities.pop(person_id, None)
        self._torso_turn_intensities.pop(person_id, None)
        self._facing.pop(person_id, None)

    def close(self):
        super().close()
        self._prev_landmarks.clear()
        self._leg_extension_intensities.clear()
        self._torso_turn_intensities.clear()
        self._facing.clear()
