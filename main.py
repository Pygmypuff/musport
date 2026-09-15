"""
Entry point for the gesture-music prototype.

YOLO detects people in the webcam feed; each person's cropped bounding
box is shown in its own window and handed off to the (currently
stubbed) MediaPipe gesture pipeline in gesture_pipeline.py.

Run:
    python main.py

Press 'q' to quit.
"""

import cv2

from gesture_pipeline import GesturePipeline
from yolo_detector import YoloPersonDetector

MODEL_PATH = "yolo26n-pose.pt"
MAIN_WINDOW = "Gesture Music Prototype"


def person_window_name(person_id: int) -> str:
    return f"Person {person_id}"


def main():
    detector = YoloPersonDetector(MODEL_PATH)
    pipeline = GesturePipeline()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    open_person_windows = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            people = detector.detect(frame)

            current_windows = set()
            for person in people:
                x1, y1, x2, y2 = person["bbox"]
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # ----------------------------------------------------------
                # MEDIAPIPE PIPELINE HOOK
                # Each person's cropped frame is handed off here so gesture
                # detection can run per-person once it's wired up.
                # ----------------------------------------------------------
                pipeline.process(person["id"], person_crop)

                window_name = person_window_name(person["id"])
                cv2.imshow(window_name, person_crop)
                current_windows.add(window_name)

            # Close windows for people no longer in frame.
            for stale_window in open_person_windows - current_windows:
                cv2.destroyWindow(stale_window)
            open_person_windows = current_windows

            annotated_frame = detector.annotate(frame)
            cv2.imshow(MAIN_WINDOW, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
