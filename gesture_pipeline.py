"""
Per-person movement handler.

Runs every movement detector over one person's cropped frame and hands
the combined results back to the caller. Adding a new kind of movement
means adding a detector here, not changing main.
"""

from hand_detector import HandDetector
from pose_detector import PoseDetector


class GesturePipeline:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()

    def process(self, person_id: int, person_crop):
        """Return (poses, gestures) detected for one person.

        `person_crop` is a BGR numpy array (frame[y1:y2, x1:x2]). Both
        lists are empty when nothing is recognized.
        """
        # Pose runs first: the hand detector draws its skeleton onto the crop,
        # and the pose model should see the unmarked image.
        poses = self.pose_detector.detect(person_id, person_crop)
        gestures = self.hand_detector.detect(person_id, person_crop)

        return poses, gestures

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()
