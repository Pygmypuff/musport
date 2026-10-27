# Musport

Nature sounds that respond to yoga movements in a senior-living exercise class.

The forest bed stays on in the background. When someone in frame does a mapped movement, the matching sound plays while they hold it (or plays once when they start it). Doing a sequence of movements in order can play a sound too.

## Movement → sound

| Sound   | Movement                         | Playback    | How it is detected |
|---------|----------------------------------|-------------|--------------------|
| Breeze  | Side bend (one hand up)          | Continuous  | One wrist above nose + shoulder tilt or outboard arm |
| Birds   | Neck movements / head turn       | One-shot    | Nose off shoulder midline, ear depth, or head motion |
| Rain    | Bend down                        | Continuous  | Nose dropped toward hips along the torso |
| Wind    | Leg extension                    | Continuous  | Straight knee angle + long hip→ankle |
| Water   | Torso turn                       | Continuous  | Shoulder line twisted vs hips, or shoulder depth |
| Leaves  | Seated heel raise                | One-shot    | Heel above toes, or ankle lifting across frames |
| Cricket | Finger extension                 | One-shot    | All four fingers extended (tip farther than pip) |

- **Continuous:** plays while anyone is doing the movement, then fades out over 0.5s and stops once nobody is.
- **One-shot:** plays once when someone starts the movement.

Sound pairing: `movements/mapping.py`. Detection thresholds: `movements/parameters.py`.

## Sequences

A sequence is a list of movements one person does in order, within a time window. Other movements in between are fine. Defined in `movements/sequences.py`:

| Sound  | Sequence              | Within |
|--------|-----------------------|--------|
| Leaves | Side bend → bend down | 5s     |

The person's tile shows the sequence name for a few seconds after they complete it.

## How sounds behave

- **Per person:** each tracked person has their own movement history, so one person's side bend and another person's bend down don't add up to a sequence.
- **No flicker:** a movement counts as started after it's been seen for 0.2s, and as ended after it's been missing for 0.4s. Both are set in `movements/parameters.py`.
- **Room-wide sounds:** sounds aren't per person. Two people side-bending give one breeze.
- **Never doubled:** each sound has one channel. A one-shot or sequence sound that's already playing ignores new triggers and plays to the end.

## Run

```bash
# Webcam + movement detection + sounds
python main.py

# Keyboard-only sound audition (no camera)
python demo_sounds.py
```

`demo_sounds.py` keys: `B` breeze, `1` birds, `2` rain, `3` wind, `4` water, `5` leaves, `C` cricket, `Q` quit. Hold continuous keys to keep them playing.

## Project structure

Each webcam frame flows through the same steps:

```
webcam frame → find people → crop each person → find body/hand landmarks
             → apply movement rules → track each person over time
             → play sounds → draw the window
```

```
main.py                 The webcam loop; ties everything below together
ui.py                   Draws the window (webcam feed + grid of person tiles)
demo_sounds.py          Keyboard sound test, no camera needed

detection/
  yolo_detector.py      Finds and tracks people (ID + bounding box)
  gesture_pipeline.py   Runs the pose and hand detectors on one person
  landmark_detector.py  Shared MediaPipe setup used by both detectors
  pose_detector.py      Body landmarks → movements
  hand_detector.py      Hand landmarks → finger extension
  person_tracker.py     Per-person history: movement start/end, sequences
movements/
  mapping.py            The sounds, and which movement plays which
  sequences.py          Movement sequences and the sound each plays
  detectors.py          The geometry rule for each movement
  parameters.py         Thresholds those rules use, plus timing
audio/
  sound_engine.py       One channel per sound: held loops, one-shot triggers

assets/sounds/          Forest wav files
Models/                 MediaPipe models (downloaded if missing)
yolo26n-pose.pt         YOLO person model
```

Where to make common changes:

- **Change a sound or add a movement:** `movements/mapping.py`, plus a rule in `movements/detectors.py`.
- **Add or change a sequence:** `movements/sequences.py`.
- **A movement triggers too often or too rarely:** `movements/parameters.py`.
- **Sounds start or stop too quickly or slowly:** the timing section of `movements/parameters.py`, or `FADE_SECONDS` in `audio/sound_engine.py`.
- **Change how the window looks:** `ui.py`.

## Tuning

If a move rarely triggers or false-triggers, edit the constants in `movements/parameters.py` (each is commented). Sit far enough back so hips and feet stay in the YOLO crop — leg extension and heel raise need visible legs.

## Next steps

1. Optionally scale continuous volume by movement depth (bend angle → intensity).
