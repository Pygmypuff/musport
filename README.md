# Musport

Nature sounds that respond to yoga movements in a senior-living exercise class.

An optional forest bed loops in the background (`BACKGROUND_SOUND_ENABLED` in `movements/enabled.py`). When someone in frame does a mapped movement, the matching sound plays while they hold it (or plays once when they start it). Doing a sequence of movements in order can play a sound too.

## Movement → sound

| Sound   | Movement                         | Playback    | How it is detected |
|---------|----------------------------------|-------------|--------------------|
| Breeze  | Side bend (one hand up)          | Continuous  | Exactly one wrist above its shoulder + shoulder elevation difference |
| Birds   | Neck movements / head turn       | One-shot    | Nose off shoulder midline, ear depth, or vertical head movement relative to shoulders |
| Rain    | Bend down                        | Continuous  | Torso axis leans at least 25° from vertical |
| Wind    | Leg extension                    | Continuous  | Knee angle above 105°; volume rises linearly to full at 180° |
| Water   | Torso turn                       | Continuous  | Shoulders turned >20° from the person's resting direction (3D shoulder depth); volume scales up to 45° |
| Leaves  | Heel raise / ankle rotation      | One-shot    | Ankle angle >120°, or ≥15° ankle rotation while the knee is extended >105° |
| Cricket | Fist → open palm → fist           | One-shot    | One person makes each hand shape in order |

- **Continuous:** plays while anyone is doing the movement, then fades out over 0.5s and stops once nobody is.
- **One-shot:** plays once when someone starts the movement.

Sound pairing: `movements/mapping.py`. Detection thresholds: `movements/parameters.py`.

## Sequences

A sequence is a list of movements one person does in order, within a time window. Other movements in between are fine. Defined in `movements/sequences.py`:

| Sound  | Sequence              | Within |
|--------|-----------------------|--------|
| Leaves | Side bend → bend down | 5s     |
| Cricket | Fist → open palm → fist | 5s |

The person's tile shows the sequence name for a few seconds after they complete it.

## Who gets analysed

People sit in a semicircle, so some face the camera side-on, where pose detection is unreliable. Everyone does the same movement, so only people facing the camera need analysing.

- Each person's **resting direction** is which way their shoulders usually face: the median over the last 30s. A torso twist barely moves it; the way their chair faces does.
- People whose resting direction is more than **40°** from the camera are skipped. They aren't run through the gesture pipeline, show as dimmed "Turned away" tiles, and are re-checked every 2s.
- Each tile's footer shows that person's angle from the camera, to help with chair placement.
- Torso turn is measured from each person's own resting direction, so someone sitting at an angle isn't always "turning".

Settings: the "Facing the camera" section of `movements/parameters.py`.

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
  facing.py             Which way each person sits; skips people turned away
movements/
  mapping.py            The sounds, and which movement plays which
  sequences.py          Movement sequences and the sound each plays
  enabled.py            On/off switches: each movement, sequence, background
  detectors.py          The geometry rule for each movement
  parameters.py         Thresholds those rules use, plus timing
audio/
  sound_engine.py       One channel per sound: held loops, one-shot triggers

assets/sounds/          Forest wav files
Models/                 MediaPipe models (downloaded if missing)
yolo26n-pose.pt         YOLO person model
```

Where to make common changes:

- **Test only some movements or sequences, or turn the background off:** set them to `True`/`False` in `movements/enabled.py`. `main.py` prints what's on at startup.
- **Change a sound or add a movement:** `movements/mapping.py`, plus a rule in `movements/detectors.py`.
- **Add or change a sequence:** `movements/sequences.py`.
- **A movement triggers too often or too rarely:** `movements/parameters.py`.
- **Sounds start or stop too quickly or slowly:** the timing section of `movements/parameters.py`, or `FADE_SECONDS` in `audio/sound_engine.py`.
- **Change how the window looks:** `ui.py`.

## Tuning

If a move rarely triggers or false-triggers, edit the constants in `movements/parameters.py` (each is commented). Sit far enough back so hips and feet stay in the YOLO crop — leg extension and heel raise need visible legs.

## Next steps

1. Optionally scale continuous volume by movement depth for other movements.
