"""
Per-person gesture pipeline — this is where MediaPipe plugs in.

"""


class GesturePipeline:
    def process(self, person_id: int, person_crop):
        """Run gesture detection on one person's cropped frame.

        `person_crop` is a BGR numpy array (frame[y1:y2, x1:x2]).
        """
        # TODO: run MediaPipe hand/pose landmark detection on person_crop
        # TODO: interpret the landmarks into a gesture
        # TODO: forward (person_id, gesture) to the music engine
        return None
