"""
Entry point for the gesture-music prototype.

YOLO detects people in the webcam feed; each person's cropped bounding
box is shown in its own window with the pose GesturePipeline detected
for them drawn on top.

Run:
    python main.py

Press 'q' to quit.
"""

import cv2

from gesture_pipeline import GesturePipeline
from pose_detector import BOTH_HANDS_UP, is_both_hands_up
from yolo_detector import YoloPersonDetector

MODEL_PATH = "yolo26n-pose.pt"
MAIN_WINDOW = "Gesture Music Prototype"

NO_LANDMARKS = "no landmarks"
NO_POSE = "no pose"

FONT = cv2.FONT_HERSHEY_SIMPLEX


def person_window_name(person_id: int) -> str:
    return f"Person {person_id}"


def pose_label(result) -> str:
    """Name the pose in a PoseLandmarkerResult from GesturePipeline."""
    if not result.pose_landmarks:
        return NO_LANDMARKS

    if is_both_hands_up(result.pose_landmarks[0]):
        return BOTH_HANDS_UP

    return NO_POSE


def draw_pose_label(crop, label):
    """Draw the pose name across the top of a person's crop."""
    width = crop.shape[1]
    color = (0, 200, 0) if label == BOTH_HANDS_UP else (0, 165, 255)

    # Person crops can be very narrow, so size the text to fit the width
    # rather than picking a fixed scale that would get clipped.
    (unit_width, _), _ = cv2.getTextSize(label, FONT, 1.0, 1)
    scale = min(0.7, (width - 8) / unit_width)
    thickness = 2 if scale >= 0.5 else 1
    (_, text_height), baseline = cv2.getTextSize(label, FONT, scale, thickness)

    # Dark strip behind the text so it stays readable over any footage.
    strip_height = text_height + baseline + 8
    cv2.rectangle(crop, (0, 0), (width, strip_height), (0, 0, 0), -1)
    cv2.putText(
        crop,
        label,
        (4, strip_height - baseline - 4),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return crop


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

                result = pipeline.process(person["id"], person_crop)
                label = pose_label(result)
                print(f"Person {person['id']}: {label}")

                # Copy first: person_crop is a view into `frame`, so drawing
                # on it would bleed into the main annotated window.
                labelled_crop = draw_pose_label(person_crop.copy(), label)

                # Window name stays fixed per person — putting the pose in the
                # name would open a new window every time the pose changed.
                window_name = person_window_name(person["id"])
                cv2.imshow(window_name, labelled_crop)
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
