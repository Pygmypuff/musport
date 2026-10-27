"""
Musport — nature sounds for senior-living yoga movements.

Webcam → YOLO people → pose/hand movements → per-person tracking
→ forest sound bed.

The window layout and drawing live in ui.py.

Run:
    python main.py

Press 'q' to quit. For keyboard-only sound testing: python demo_sounds.py
"""

import time
from pathlib import Path

import cv2

import ui
from audio.sound_engine import SoundEngine
from detection.gesture_pipeline import GesturePipeline
from detection.person_tracker import Event, PersonTracker, SequenceEvent
from detection.yolo_detector import YoloPersonDetector
from movements.mapping import MOVEMENT_SOUNDS


MODEL_PATH = str(Path(__file__).resolve().parent / "yolo26n-pose.pt")


def update_sounds(sounds: SoundEngine, tracker: PersonTracker, events: list[Event]):
    """
    Continuous movements hold their sound while anyone is doing them.
    One-shot movements and completed sequences trigger theirs once.
    """
    sounds.set_held({
        MOVEMENT_SOUNDS[m].sound
        for m in tracker.active_movements()
        if MOVEMENT_SOUNDS[m].playback == "continuous"
    })

    for event in events:
        if isinstance(event, SequenceEvent):
            sounds.trigger(event.sequence.sound)
        elif event.kind == "start":
            spec = MOVEMENT_SOUNDS[event.movement]
            if spec.playback == "oneshot":
                sounds.trigger(spec.sound)

    sounds.update()


def main():
    detector = YoloPersonDetector(MODEL_PATH)
    pipeline = GesturePipeline()
    tracker = PersonTracker(on_person_removed=pipeline.forget)
    sounds = SoundEngine(enable_background=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.monotonic()

            detections = {}
            crops = {}

            for person in detector.detect(frame):
                x1, y1, x2, y2 = person["bbox"]
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Copy because HandDetector draws landmarks onto the crop,
                # and person_crop is a view into `frame`.
                display_crop = person_crop.copy()

                detections[person["id"]] = pipeline.process(
                    person["id"], display_crop
                )
                crops[person["id"]] = display_crop

            events = tracker.update(detections, now)
            update_sounds(sounds, tracker, events)

            person_views = []
            for person_id, crop in crops.items():
                person = tracker.people[person_id]
                person_views.append((
                    person_id,
                    crop,
                    person.movements_in_progress(),
                    person.recent_sequence(now),
                ))

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
