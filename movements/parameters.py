"""
Tunable thresholds for seated senior yoga movement detection.

Values are normalized to MediaPipe landmark space (roughly 0–1 in the
person crop) or degrees. Loosen a number if a move rarely fires; tighten
if it false-triggers.
"""

# Landmark must be at least this present/visible to count.
MIN_PRESENCE = 0.45

# --- Breeze: side bend, one hand up ---------------------------------
# Wrist is above its shoulder by this much (y grows downward).
SIDE_BEND_WRIST_ABOVE_SHOULDER = 0.02
# Shoulder elevation difference |Δy| / shoulder_width.
SIDE_BEND_SHOULDER_ELEVATION = 0.12
# Side-bend score ramp start/end for the torso lean angle used by the
# revised movement scorer.
SIDE_BEND_MIN_LATERAL_DEG = 10.0
SIDE_BEND_FULL_LATERAL_DEG = 30.0
# If a single raised arm is not visible, the score is multiplied by this.
SIDE_BEND_NO_ARM_FACTOR = 0.5

# --- Birds: neck / head turn ----------------------------------------
# |nose.x − shoulder_mid.x| / shoulder_width
NECK_TURN_NOSE_OFFSET = 0.14
# Ear depth asymmetry |left_ear.z − right_ear.z|
NECK_TURN_EAR_Z = 0.08
# Temporal: nose.x jump across frames (fraction of shoulder width).
NECK_TURN_NOSE_DELTA = 0.06
# Per-frame downward nose movement relative to the shoulder midpoint,
# normalized by shoulder width.
NECK_TURN_NOSE_VERTICAL_DELTA = 0.035
# Relative head-yaw ramp used by the revised scorer.
NECK_TURN_MIN_DEG = 15.0
NECK_TURN_FULL_DEG = 40.0

# --- Rain: bend down / forward fold ---------------------------------
# Minimum angle of the hip-to-shoulder torso axis away from vertical.
BEND_DOWN_MIN_ANGLE_DEG = 25.0
BEND_DOWN_FULL_ANGLE_DEG = BEND_DOWN_MIN_ANGLE_DEG + 25.0

# --- Wind: leg extension --------------------------------------------
# Knee angle (hip–knee–ankle) must exceed this to count as extended.
LEG_EXT_MIN_KNEE_ANGLE = 105.0
# Wind reaches full volume at a straight-knee angle.
LEG_EXT_FULL_VOLUME_ANGLE = 180.0

# --- Water: torso turn ----------------------------------------------
# How far the shoulders have turned from the person's resting direction
# (see "Facing the camera" below), in degrees. Water starts here...
TORSO_TURN_MIN_ANGLE_DEG = 20.0
# ...and reaches full volume here (a full seated twist).
TORSO_TURN_FULL_VOLUME_ANGLE_DEG = 45.0

# --- Leaves: ankle angle / rotation ---------------------------------
# Ankle angle between ankle→knee and ankle→foot_index.
HEEL_RAISE_ANKLE_ANGLE_DEG = 120.0
# Ankle angle change that counts as rotation while that leg is extended.
HEEL_RAISE_ANKLE_ROTATION_DEG = 15.0
# Toe movement relative to the ankle between frames, as a fraction of
# shin length. Starting guess; tune against real footage.
HEEL_RAISE_ANKLE_MOVEMENT = 0.10

# --- Cricket: fist / open-palm sequence -----------------------------
# tip→wrist distance must exceed pip→wrist by this factor.
FINGER_EXT_TIP_RATIO = 1.08
# How many of index/middle/ring/pinky must be extended (of 4).
FINGER_EXT_MIN_COUNT = 4
# All four fingertips must be this much closer to the wrist than their PIP joints.
FIST_TIP_RATIO_MAX = 0.95
# Revised scorer thresholds for hand posture.
FINGER_EXT_MIN_PIP_ANGLE_DEG = 150.0
FIST_MAX_PIP_ANGLE_DEG = 100.0

# --- Heel raise / ankle motion ---------------------------------------
# Window over which the ankle flexion range is measured.
HEEL_RAISE_WINDOW_S = 0.6
HEEL_RAISE_MIN_RANGE_DEG = 10.0
HEEL_RAISE_FULL_RANGE_DEG = 30.0

# --- Facing the camera: who gets analysed ---------------------------
# A person's resting direction is the median of their shoulder direction
# over this many seconds. Torso turns held longer than about half of
# this start to count as the new resting direction and fade out.
RESTING_DIRECTION_SECONDS = 30.0
# People whose resting direction is turned further than this from the
# camera are skipped: not run through the gesture pipeline at all.
FACING_MAX_TURN_DEG = 40.0
# ...and are let back in once they're this many degrees inside the limit
# (stops someone right at the limit flickering in and out).
FACING_HYSTERESIS_DEG = 5.0
# Frames needed before a person's direction is known. Until then they're
# analysed, but torso turn can't fire yet.
FACING_MIN_SAMPLES = 5
# How often a skipped person is re-measured, in case they've moved.
FACING_RECHECK_SECONDS = 2.0

# --- Timing (seconds) -----------------------------------------------
# A movement must be detected for this long before it counts as started.
MOVEMENT_START_SECONDS = 0.2
# ...and missing for this long before it counts as ended. Short gaps
# (a dropped frame, a brief occlusion) shorter than this are ignored.
MOVEMENT_END_SECONDS = 0.4
# Forget a person who has been out of frame this long.
PERSON_TIMEOUT_SECONDS = 2.0
# How far back each person's movement history goes. Must be at least
# as long as the longest sequence window in movements/sequences.py.
HISTORY_SECONDS = 30.0
