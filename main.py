"""
Musport — nature sounds for senior-living yoga movements.

Webcam → YOLO people → pose/hand movements → per-person tracking
→ the person sitting straightest controls the forest sound bed.

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
from detection.facing import ControllerPicker
from detection.gesture_pipeline import GesturePipeline
from detection.person_tracker import Event, PersonTracker, SequenceEvent
from detection.yolo_detector import YoloPersonDetector
from movements import enabled
from movements.mapping import MOVEMENT_SOUNDS, Movement


MODEL_PATH = str(Path(__file__).resolve().parent / "yolo26n-pose.pt")


def update_sounds(
    sounds: SoundEngine,
    tracker: PersonTracker,
    events: list[Event],
    movement_intensities: dict[Movement, dict[int, float]],
    controller_id: int | None,
):
    """
    Only the controller's movements make sound; everyone else is ignored.
    Their continuous movements hold their sound while they do them;
    one-shot movements and completed sequences trigger theirs once.
    """
    controller = tracker.people.get(controller_id)
    active = controller.active if controller else set()

    held_sounds = {
        MOVEMENT_SOUNDS[m].sound
        for m in active
        if MOVEMENT_SOUNDS[m].playback == "continuous"
    }
    volume_scales = {
        MOVEMENT_SOUNDS[movement].sound: (
            person_intensities.get(controller_id, 0.0)
            if movement in active
            else 0.0
        )
        for movement, person_intensities in movement_intensities.items()
    }
    sounds.set_held(held_sounds, volume_scales=volume_scales)

    for event in events:
        if event.person_id != controller_id:
            continue
        if isinstance(event, SequenceEvent):
            sounds.trigger(event.sequence.sound)
        elif event.kind == "start":
            spec = MOVEMENT_SOUNDS[event.movement]
            if spec.playback == "oneshot":
                sounds.trigger(spec.sound)

    sounds.update()


def main():
    print(enabled.summary())

    detector = YoloPersonDetector(MODEL_PATH)
    pipeline = GesturePipeline()
    tracker = PersonTracker(on_person_removed=pipeline.forget)
    controller_picker = ControllerPicker()
    sounds = SoundEngine()

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
            movement_intensities = {
                Movement.LEG_EXTENSION: {},
                Movement.TORSO_TURN: {},
            }

            for person in detector.detect(frame):
                x1, y1, x2, y2 = person["bbox"]
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Copy because HandDetector draws landmarks onto the crop,
                # and person_crop is a view into `frame`.
                display_crop = person_crop.copy()

                detections[person["id"]] = pipeline.process(
                    person["id"], display_crop, now
                )
                movement_intensities[Movement.LEG_EXTENSION][person["id"]] = (
                    pipeline.leg_extension_intensity(person["id"])
                )
                movement_intensities[Movement.TORSO_TURN][person["id"]] = (
                    pipeline.torso_turn_intensity(person["id"])
                )
                crops[person["id"]] = display_crop

            events = tracker.update(detections, now)
            controller_id = controller_picker.update({
                person_id: pipeline.angle_from_camera(person_id)
                for person_id in tracker.people
            })
            update_sounds(
                sounds, tracker, events, movement_intensities, controller_id
            )

            person_views = []
            for person_id, crop in crops.items():
                person = tracker.people[person_id]
                person_views.append(ui.PersonView(
                    person_id=person_id,
                    crop=crop,
                    movements=person.movements_in_progress(),
                    sequence=person.recent_sequence(now),
                    angle_from_camera=pipeline.angle_from_camera(person_id),
                    controls_sound=person_id == controller_id,
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
