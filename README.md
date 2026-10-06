# Musport

Nature sounds that respond to yoga movements in a senior-living exercise class.

The forest bed stays on in the background. When someone in frame does a mapped movement, the matching sound fades in (or plays once).

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

Sound pairing: `movements/mapping.py`. Detection thresholds: `movements/parameters.py`.

## Run

```bash
# Webcam + movement detection + sounds
python main.py

# Keyboard-only sound audition (no camera)
python demo_sounds.py
```

`demo_sounds.py` keys: `B` breeze, `1` birds, `2` rain, `3` wind, `4` water, `5` leaves, `C` cricket, `Q` quit. Hold continuous keys to keep them playing.

## Layout

```
main.py                 Webcam app (YOLO → movements → sounds)
demo_sounds.py          Keyboard sound test
gesture_pipeline.py     Combines pose + hand detectors
pose_detector.py        Body pose → Movement IDs
hand_detector.py        Hands → finger extension
yolo_detector.py        Person bounding boxes
movements/
  mapping.py            Canonical movement ↔ sound map
  parameters.py         Tunable detection thresholds
  detectors.py          Landmark geometry rules
audio/
  sound_engine.py       Pygame mixer / fade / triggers
assets/sounds/          Forest wav files
Models/                 MediaPipe .task models
```

## Tuning

If a move rarely triggers or false-triggers, edit the constants in `movements/parameters.py` (each is commented). Sit far enough back so hips and feet stay in the YOLO crop — leg extension and heel raise need visible legs.

## Next steps

1. Soften detection (hold time / hysteresis) so sounds don’t flicker in a class setting.
2. Optionally scale continuous volume by movement depth (bend angle → intensity).
3. Decide whether one person’s movement drives the whole room, or sounds layer per person.
