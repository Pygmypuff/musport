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

from movements import parameters as P
from movements.mapping import Movement

LM = PoseLandmark


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


def _mid(a: tuple[float, float, float], b: tuple[float, float, float]):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _dist2(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a, b, c) -> float:
    """Angle ABC in degrees (at point b)."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return math.degrees(math.acos(cos_a))


def _line_angle_deg(a, b, c, d) -> float:
    """Absolute angle between vectors ab and cd, in degrees [0, 90]."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = d[0] - c[0], d[1] - c[1]
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, abs(ux * vx + uy * vy) / (nu * nv)))
    # Angle between lines: 0 = parallel, 90 = perpendicular.
    return math.degrees(math.acos(cos_a))


def detect_side_bend(landmarks, prev_landmarks=None) -> bool:
    """Breeze — one hand raised with a side lean."""
    needed = (
        LM.NOSE,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_WRIST,
        LM.RIGHT_WRIST,
    )
    if not _present(landmarks, *needed):
        return False

    nose = _pt(landmarks, LM.NOSE)
    l_sh = _pt(landmarks, LM.LEFT_SHOULDER)
    r_sh = _pt(landmarks, LM.RIGHT_SHOULDER)
    l_wr = _pt(landmarks, LM.LEFT_WRIST)
    r_wr = _pt(landmarks, LM.RIGHT_WRIST)

    shoulder_w = max(_dist2(l_sh, r_sh), 1e-3)
    tilt = abs(l_sh[1] - r_sh[1]) / shoulder_w

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
    needed = (LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if not _present(landmarks, *needed):
        return False

    nose = _pt(landmarks, LM.NOSE)
    l_sh = _pt(landmarks, LM.LEFT_SHOULDER)
    r_sh = _pt(landmarks, LM.RIGHT_SHOULDER)
    mid = _mid(l_sh, r_sh)
    shoulder_w = max(_dist2(l_sh, r_sh), 1e-3)

    nose_offset = abs(nose[0] - mid[0]) / shoulder_w
    if nose_offset >= P.NECK_TURN_NOSE_OFFSET:
        return True

    if _present(landmarks, LM.LEFT_EAR, LM.RIGHT_EAR):
        l_ear = _pt(landmarks, LM.LEFT_EAR)
        r_ear = _pt(landmarks, LM.RIGHT_EAR)
        if abs(l_ear[2] - r_ear[2]) >= P.NECK_TURN_EAR_Z:
            return True

    if prev_landmarks is not None and _present(prev_landmarks, LM.NOSE):
        prev_nose = _pt(prev_landmarks, LM.NOSE)
        if abs(nose[0] - prev_nose[0]) / shoulder_w >= P.NECK_TURN_NOSE_DELTA:
            return True

    return False


def detect_bend_down(landmarks, prev_landmarks=None) -> bool:
    """Rain — forward fold / bend down (seated or standing)."""
    needed = (
        LM.NOSE,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if not _present(landmarks, *needed):
        return False

    nose = _pt(landmarks, LM.NOSE)
    sh = _mid(_pt(landmarks, LM.LEFT_SHOULDER), _pt(landmarks, LM.RIGHT_SHOULDER))
    hip = _mid(_pt(landmarks, LM.LEFT_HIP), _pt(landmarks, LM.RIGHT_HIP))

    torso = hip[1] - sh[1]
    if torso < 0.05:
        return False  # hips not below shoulders — bad crop / lying down

    # Nose traveled from shoulders toward hips (~0 upright → ~1 folded).
    nose_ratio = (nose[1] - sh[1]) / torso
    if nose_ratio >= P.BEND_DOWN_NOSE_RATIO:
        return True

    # Forward lean toward camera shortens apparent torso vs hip width.
    hip_w = max(
        _dist2(_pt(landmarks, LM.LEFT_HIP), _pt(landmarks, LM.RIGHT_HIP)), 1e-3
    )
    if torso / hip_w < P.BEND_DOWN_TORSO_COMPRESS and nose[1] > sh[1] + 0.05:
        return True

    return False


def _leg_extended(landmarks, hip_i, knee_i, ankle_i) -> bool:
    if not _present(landmarks, hip_i, knee_i, ankle_i):
        return False
    hip = _pt(landmarks, hip_i)
    knee = _pt(landmarks, knee_i)
    ankle = _pt(landmarks, ankle_i)

    angle = _angle_deg(hip, knee, ankle)
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
    needed = (
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if not _present(landmarks, *needed):
        return False

    l_sh = _pt(landmarks, LM.LEFT_SHOULDER)
    r_sh = _pt(landmarks, LM.RIGHT_SHOULDER)
    l_hip = _pt(landmarks, LM.LEFT_HIP)
    r_hip = _pt(landmarks, LM.RIGHT_HIP)

    # Twist: shoulder line rotated vs hip line.
    twist = _line_angle_deg(l_sh, r_sh, l_hip, r_hip)
    if twist >= P.TORSO_TURN_LINE_ANGLE:
        return True

    if abs(l_sh[2] - r_sh[2]) >= P.TORSO_TURN_SHOULDER_Z:
        return True

    return False


def detect_heel_raise(landmarks, prev_landmarks=None) -> bool:
    """Leaves — seated heel raise (heels up, toes down)."""

    def side(heel_i, toe_i, ankle_i) -> bool:
        if _present(landmarks, heel_i, toe_i):
            heel = _pt(landmarks, heel_i)
            toe = _pt(landmarks, toe_i)
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


def detect_finger_extension(hand_landmarks: Sequence) -> bool:
    """Cricket — open hand / fingers extended."""
    wrist = hand_landmarks[0]

    def extended(tip: int, pip: int) -> bool:
        tip_d = math.hypot(
            hand_landmarks[tip].x - wrist.x, hand_landmarks[tip].y - wrist.y
        )
        pip_d = math.hypot(
            hand_landmarks[pip].x - wrist.x, hand_landmarks[pip].y - wrist.y
        )
        if pip_d < 1e-6:
            return False
        return tip_d / pip_d >= P.FINGER_EXT_TIP_RATIO

    fingers = (
        extended(8, 6),
        extended(12, 10),
        extended(16, 14),
        extended(20, 18),
    )
    return sum(fingers) >= P.FINGER_EXT_MIN_COUNT


# Pose-landmark detectors run by PoseDetector, in priority order.
POSE_DETECTORS: list[tuple[Movement, Callable]] = [
    (Movement.SIDE_BEND, detect_side_bend),
    (Movement.NECK_TURN, detect_neck_turn),
    (Movement.BEND_DOWN, detect_bend_down),
    (Movement.LEG_EXTENSION, detect_leg_extension),
    (Movement.TORSO_TURN, detect_torso_turn),
    (Movement.HEEL_RAISE, detect_heel_raise),
]
