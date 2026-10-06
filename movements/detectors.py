"""
Pose / hand landmark → yoga Movement IDs.

Each detector is a geometric rule tuned for seated senior classes.
Thresholds live in movements/parameters.py so they can be adjusted
without rewriting the math.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from mediapipe.tasks.python.vision import PoseLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from movements import parameters as P
from movements.mapping import Movement

LM = PoseLandmark
HAND = HandLandmark

# (tip, pip) joints of the four non-thumb fingers.
FINGER_JOINTS = (
    (HAND.INDEX_FINGER_TIP, HAND.INDEX_FINGER_PIP),
    (HAND.MIDDLE_FINGER_TIP, HAND.MIDDLE_FINGER_PIP),
    (HAND.RING_FINGER_TIP, HAND.RING_FINGER_PIP),
    (HAND.PINKY_TIP, HAND.PINKY_PIP),
)


# --- Landmark access -------------------------------------------------


def _vis(lm) -> float:
    """Best available confidence for a landmark."""
    presence = getattr(lm, "presence", None)
    if presence is not None and presence > 0:
        return float(presence)
    return float(getattr(lm, "visibility", 0.0) or 0.0)


def _present(landmarks, *indices) -> bool:
    return all(_vis(landmarks[i]) >= P.MIN_PRESENCE for i in indices)


def _pt(landmarks, index) -> tuple[float, float, float]:
    lm = landmarks[index]
    return float(lm.x), float(lm.y), float(lm.z)


def _points(landmarks, *indices):
    """(x, y, z) for each landmark, or None unless all of them are present."""
    if not _present(landmarks, *indices):
        return None
    return [_pt(landmarks, i) for i in indices]


# --- Geometry (x/y image plane unless noted) -------------------------


def _mid(a: tuple[float, float, float], b: tuple[float, float, float]):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _dist2(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _width(a, b) -> float:
    """Distance between two points, floored so it is safe to divide by."""
    return max(_dist2(a, b), 1e-3)


def _vec(a, b) -> tuple[float, float]:
    """Vector from a to b."""
    return b[0] - a[0], b[1] - a[1]


def _angle_deg(u, v, as_lines: bool = False) -> float:
    """Angle between vectors u and v in degrees [0, 180].

    With as_lines=True direction is ignored, giving the angle between
    the two lines instead: 0 = parallel, 90 = perpendicular.
    """
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_a = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    if as_lines:
        cos_a = abs(cos_a)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


# --- Pose detectors --------------------------------------------------


def detect_side_bend(landmarks, prev_landmarks=None) -> bool:
    """Breeze — one hand raised with a side lean."""
    points = _points(
        landmarks,
        LM.NOSE,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_WRIST,
        LM.RIGHT_WRIST,
    )
    if points is None:
        return False
    nose, l_sh, r_sh, l_wr, r_wr = points

    tilt = abs(l_sh[1] - r_sh[1]) / _width(l_sh, r_sh)

    left_up = l_wr[1] < nose[1] - P.SIDE_BEND_WRIST_ABOVE_NOSE
    right_up = r_wr[1] < nose[1] - P.SIDE_BEND_WRIST_ABOVE_NOSE
    if left_up == right_up:
        return False  # need exactly one hand up

    if left_up:
        outboard = l_wr[0] > l_sh[0] + P.SIDE_BEND_WRIST_OUTBOARD
    else:
        outboard = r_wr[0] < r_sh[0] - P.SIDE_BEND_WRIST_OUTBOARD

    # One hand up is the class cue; tilt or outboard arm confirms side bend.
    return tilt >= P.SIDE_BEND_SHOULDER_TILT or outboard


def detect_neck_turn(landmarks, prev_landmarks=None) -> bool:
    """Birds — head / neck turned left or right."""
    points = _points(landmarks, LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if points is None:
        return False
    nose, l_sh, r_sh = points

    mid = _mid(l_sh, r_sh)
    shoulder_w = _width(l_sh, r_sh)

    nose_offset = abs(nose[0] - mid[0]) / shoulder_w
    if nose_offset >= P.NECK_TURN_NOSE_OFFSET:
        return True

    ears = _points(landmarks, LM.LEFT_EAR, LM.RIGHT_EAR)
    if ears is not None:
        l_ear, r_ear = ears
        if abs(l_ear[2] - r_ear[2]) >= P.NECK_TURN_EAR_Z:
            return True

    if prev_landmarks is not None and _present(prev_landmarks, LM.NOSE):
        prev_nose = _pt(prev_landmarks, LM.NOSE)
        if abs(nose[0] - prev_nose[0]) / shoulder_w >= P.NECK_TURN_NOSE_DELTA:
            return True

    return False


def detect_bend_down(landmarks, prev_landmarks=None) -> bool:
    """Rain — forward fold / bend down (seated or standing)."""
    points = _points(
        landmarks,
        LM.NOSE,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if points is None:
        return False
    nose, l_sh, r_sh, l_hip, r_hip = points

    sh = _mid(l_sh, r_sh)
    hip = _mid(l_hip, r_hip)

    torso = hip[1] - sh[1]
    if torso < 0.05:
        return False  # hips not below shoulders — bad crop / lying down

    # Nose traveled from shoulders toward hips (~0 upright → ~1 folded).
    nose_ratio = (nose[1] - sh[1]) / torso
    if nose_ratio >= P.BEND_DOWN_NOSE_RATIO:
        return True

    # Forward lean toward camera shortens apparent torso vs hip width.
    hip_w = _width(l_hip, r_hip)
    if torso / hip_w < P.BEND_DOWN_TORSO_COMPRESS and nose[1] > sh[1] + 0.05:
        return True

    return False


def _leg_extended(landmarks, hip_i, knee_i, ankle_i) -> bool:
    points = _points(landmarks, hip_i, knee_i, ankle_i)
    if points is None:
        return False
    hip, knee, ankle = points

    # Knee angle (hip–knee–ankle); 180° is straight.
    angle = _angle_deg(_vec(knee, hip), _vec(knee, ankle))
    if angle < P.LEG_EXT_MIN_KNEE_ANGLE:
        return False

    if _dist2(hip, ankle) < P.LEG_EXT_MIN_HIP_ANKLE:
        return False

    # Seated kick: ankle roughly level with knee (not hanging straight down).
    if abs(ankle[1] - knee[1]) > P.LEG_EXT_ANKLE_KNEE_Y:
        # Still allow standing-ish straight leg if hip–ankle is long and knee open.
        if angle < 165.0:
            return False

    return True


def detect_leg_extension(landmarks, prev_landmarks=None) -> bool:
    """Wind — one or both legs extended."""
    left = _leg_extended(landmarks, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)
    right = _leg_extended(landmarks, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)
    return left or right


def detect_torso_turn(landmarks, prev_landmarks=None) -> bool:
    """Water — seated torso rotation."""
    points = _points(
        landmarks,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if points is None:
        return False
    l_sh, r_sh, l_hip, r_hip = points

    # Twist: shoulder line rotated vs hip line.
    twist = _angle_deg(_vec(l_sh, r_sh), _vec(l_hip, r_hip), as_lines=True)
    if twist >= P.TORSO_TURN_LINE_ANGLE:
        return True

    if abs(l_sh[2] - r_sh[2]) >= P.TORSO_TURN_SHOULDER_Z:
        return True

    return False


def detect_heel_raise(landmarks, prev_landmarks=None) -> bool:
    """Leaves — seated heel raise (heels up, toes down)."""

    def side(heel_i, toe_i, ankle_i) -> bool:
        foot = _points(landmarks, heel_i, toe_i)
        if foot is not None:
            heel, toe = foot
            if toe[1] - heel[1] >= P.HEEL_RAISE_HEEL_ABOVE_TOES:
                return True

        if (
            prev_landmarks is not None
            and _present(landmarks, ankle_i)
            and _present(prev_landmarks, ankle_i)
        ):
            ankle = _pt(landmarks, ankle_i)
            prev = _pt(prev_landmarks, ankle_i)
            # y decreases as the ankle rises in the image.
            if prev[1] - ankle[1] >= P.HEEL_RAISE_ANKLE_LIFT:
                return True
        return False

    left = side(LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX, LM.LEFT_ANKLE)
    right = side(LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX, LM.RIGHT_ANKLE)
    return left or right


# --- Hand detectors --------------------------------------------------


def detect_finger_extension(hand_landmarks: Sequence) -> bool:
    """Cricket — open hand / fingers extended."""
    wrist = _pt(hand_landmarks, HAND.WRIST)

    def extended(tip: int, pip: int) -> bool:
        tip_d = _dist2(_pt(hand_landmarks, tip), wrist)
        pip_d = _dist2(_pt(hand_landmarks, pip), wrist)
        if pip_d < 1e-6:
            return False
        return tip_d / pip_d >= P.FINGER_EXT_TIP_RATIO

    extended_count = sum(extended(tip, pip) for tip, pip in FINGER_JOINTS)
    return extended_count >= P.FINGER_EXT_MIN_COUNT


# Pose-landmark detectors run by PoseDetector, in priority order.
POSE_DETECTORS: list[tuple[Movement, Callable]] = [
    (Movement.SIDE_BEND, detect_side_bend),
    (Movement.NECK_TURN, detect_neck_turn),
    (Movement.BEND_DOWN, detect_bend_down),
    (Movement.LEG_EXTENSION, detect_leg_extension),
    (Movement.TORSO_TURN, detect_torso_turn),
    (Movement.HEEL_RAISE, detect_heel_raise),
]
