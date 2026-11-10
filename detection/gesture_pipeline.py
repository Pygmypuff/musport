"""
Per-person movement handler.

Runs pose + hand detectors on one cropped person frame and returns the
combined Movement list. Add new detectors in movements/detectors.py;
main only needs to react to Movement IDs.

People turned too far away from the camera are skipped (see
detection/facing.py): process() returns None for them.
"""

from detection.hand_detector import HandDetector
from detection.pose_detector import PoseDetector
from movements.enabled import is_enabled
from movements.mapping import Movement


class GesturePipeline:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()

    def process(
        self, person_id: int, person_crop, now: float
    ) -> list[Movement] | None:
        """Return all yoga movements detected for one person, or None if
        they're turned too far from the camera to analyse.

        Pose runs first: the hand detector draws its skeleton onto the
        crop, and the pose model should see the unmarked image. It also
        measures which way the person faces.
        """
        facing = self.pose_detector.facing(person_id)
        if facing.should_skip(now):
            return None

        # A model whose movements are all switched off isn't run at all.
        poses, hands = [], []
        if any(map(is_enabled, PoseDetector.MOVEMENTS)):
            poses = self.pose_detector.detect(person_id, person_crop, now)
        if facing.turned_away:
            facing.skipped(now)
            return None
        if any(map(is_enabled, HandDetector.MOVEMENTS)):
            hands = self.hand_detector.detect(person_id, person_crop)
        # Preserve order, drop duplicates.
        seen: set[Movement] = set()
        combined: list[Movement] = []
        for movement in poses + hands:
            if movement not in seen:
                seen.add(movement)
                combined.append(movement)
        return combined

    def angle_from_camera(self, person_id: int) -> float | None:
        """How far this person's resting direction is from the camera."""
        return self.pose_detector.facing(person_id).angle_from_camera

    def leg_extension_intensity(self, person_id: int) -> float:
        return self.pose_detector.leg_extension_intensity(person_id)

    def torso_turn_intensity(self, person_id: int) -> float:
        return self.pose_detector.torso_turn_intensity(person_id)

    def forget(self, person_id: int):
        """Drop any per-person state kept for someone who has left."""
        self.pose_detector.forget(person_id)

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
