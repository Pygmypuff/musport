"""
Musport — nature sounds for senior-living yoga movements.

Webcam → YOLO people → pose/hand movements → forest sound bed.

The window layout and drawing live in ui.py.

Run:
    python main.py

Press 'q' to quit. For keyboard-only sound testing: python demo_sounds.py
"""

from pathlib import Path

import cv2

import ui
from audio.sound_engine import SoundEngine
from gesture_pipeline import GesturePipeline
from movements.mapping import Movement
from yolo_detector import YoloPersonDetector


MODEL_PATH = str(Path(__file__).resolve().parent / "yolo26n-pose.pt")


def main():
    detector = YoloPersonDetector(MODEL_PATH)
    pipeline = GesturePipeline()
    sounds = SoundEngine(enable_background=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            person_views = []
            active_movements: set[Movement] = set()

            for person in detector.detect(frame):
                x1, y1, x2, y2 = person["bbox"]
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Copy because HandDetector draws landmarks onto the crop,
                # and person_crop is a view into `frame`.
                display_crop = person_crop.copy()

                movements = pipeline.process(person["id"], display_crop)
                active_movements.update(movements)
                person_views.append((person["id"], display_crop, movements))

            sounds.set_active(active_movements)
            sounds.update()

            cv2.imshow(
                ui.WINDOW_NAME,
                ui.render(detector.annotate(frame), person_views),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sounds.close()
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
