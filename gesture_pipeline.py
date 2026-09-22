"""
Per-person gesture pipeline — this is where MediaPipe plugs in.

"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
class GesturePipeline:
    # Set up mediapipe model plus its options.
    def __init__(self):
        self.model_path = "Models/pose_landmarker_full.task"
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker

        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options = self.BaseOptions(model_asset_path=self.model_path),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence = .7,
            min_tracking_confidence = .7
        )

        # Create object that will track poses.
        self.pose_landmarker = self.PoseLandmarker.create_from_options(self.options)
        # Tracks frames for mediapipe to properly track movement.
        self.prev_time = 0
        self.frame_timestamp_ms = 0

    def process(self, person_id: int, person_crop):
        """Run gesture detection on one person's cropped frame.

        `person_crop` is a BGR numpy array (frame[y1:y2, x1:x2]).
        """
        # TODO: run MediaPipe hand/pose landmark detection on person_crop
        # TODO: interpret the landmarks into a gesture
        # TODO: forward (person_id, gesture) to the music engine
        
        
        #  Converts frame from BGR to RGB which is what mediapipie uses.
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_timestamp_ms += 33
        result = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        return result
