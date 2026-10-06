"""
Tunable thresholds for seated senior yoga movement detection.

Values are normalized to MediaPipe landmark space (roughly 0–1 in the
person crop) or degrees. Loosen a number if a move rarely fires; tighten
if it false-triggers.
"""

# Landmark must be at least this present/visible to count.
MIN_PRESENCE = 0.45

# --- Breeze: side bend, one hand up ---------------------------------
# Wrist is "up" when its y is this far above the nose (y grows downward).
SIDE_BEND_WRIST_ABOVE_NOSE = 0.02
# Shoulder line tilt |Δy| / shoulder_width — torso lean cue.
SIDE_BEND_SHOULDER_TILT = 0.12
# Raised wrist should sit outward past the shoulder on that side.
SIDE_BEND_WRIST_OUTBOARD = 0.02

# --- Birds: neck / head turn ----------------------------------------
# |nose.x − shoulder_mid.x| / shoulder_width
NECK_TURN_NOSE_OFFSET = 0.18
# Ear depth asymmetry |left_ear.z − right_ear.z|
NECK_TURN_EAR_Z = 0.08
# Temporal: nose.x jump across frames (fraction of shoulder width).
NECK_TURN_NOSE_DELTA = 0.06

# --- Rain: bend down / forward fold ---------------------------------
# How far the nose has traveled from shoulders toward hips:
#   (nose.y − shoulder_mid.y) / (hip_mid.y − shoulder_mid.y)
# Negative when upright (nose above shoulders); rises as you fold.
BEND_DOWN_NOSE_RATIO = 0.20
# Apparent torso height / hip width; smaller ⇒ leaning toward camera.
BEND_DOWN_TORSO_COMPRESS = 0.85

# --- Wind: leg extension --------------------------------------------
# Knee angle (hip–knee–ankle) in degrees; 180° is straight.
LEG_EXT_MIN_KNEE_ANGLE = 155.0
# Extended ankle should be farther from hip than the bent threshold.
LEG_EXT_MIN_HIP_ANKLE = 0.22
# Prefer the extended foot roughly level with the knee (seated kick).
LEG_EXT_ANKLE_KNEE_Y = 0.12

# --- Water: torso turn ----------------------------------------------
# Absolute angle between shoulder line and hip line (degrees).
TORSO_TURN_LINE_ANGLE = 18.0
# Shoulder depth asymmetry |z_left − z_right|
TORSO_TURN_SHOULDER_Z = 0.12

# --- Leaves: seated heel raise --------------------------------------
# Heel raised above toes: foot_index.y − heel.y (y down).
HEEL_RAISE_HEEL_ABOVE_TOES = 0.02
# Temporal: ankle y decreased (moved up) vs previous frame.
HEEL_RAISE_ANKLE_LIFT = 0.015

# --- Cricket: finger extension --------------------------------------
# tip→wrist distance must exceed pip→wrist by this factor.
FINGER_EXT_TIP_RATIO = 1.08
# How many of index/middle/ring/pinky must be extended (of 4).
FINGER_EXT_MIN_COUNT = 4
