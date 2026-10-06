"""
Per-person movement handler.

Runs pose + hand detectors on one cropped person frame and returns the
combined Movement list. Add new detectors in movements/detectors.py;
main only needs to react to Movement IDs.
"""

from detection.hand_detector import HandDetector
from detection.pose_detector import PoseDetector
from movements.mapping import Movement


class GesturePipeline:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()

    def process(self, person_id: int, person_crop) -> list[Movement]:
        """Return all yoga movements detected for one person.

        Pose runs first: the hand detector draws its skeleton onto the
        crop, and the pose model should see the unmarked image.
        """
        poses = self.pose_detector.detect(person_id, person_crop)
        hands = self.hand_detector.detect(person_id, person_crop)
        # Preserve order, drop duplicates.
        seen: set[Movement] = set()
        combined: list[Movement] = []
        for movement in poses + hands:
            if movement not in seen:
                seen.add(movement)
                combined.append(movement)
        return combined

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
