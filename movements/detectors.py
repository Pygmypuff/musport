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


def _vec(a, b, dims: int = 2) -> tuple[float, ...]:
    """Vector from a to b using the first `dims` coordinates (2 = x/y, 3 = x/y/z)."""
    return tuple(b[i] - a[i] for i in range(dims))


def _angle_deg(u, v, as_lines: bool = False) -> float:
    """Angle between vectors u and v in degrees [0, 180].

    Works for vectors of any matching length (2D or 3D).
    With as_lines=True direction is ignored, giving the angle between
    the two lines instead: 0 = parallel, 90 = perpendicular.
    """
    nu = math.hypot(*u)
    nv = math.hypot(*v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_a = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    if as_lines:
        cos_a = abs(cos_a)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))

def _dist3(a, b) -> float:
    """3D distance between two (x, y, z) points."""
    return math.dist(a[:3], b[:3])

# --- Pose detectors --------------------------------------------------


def detect_side_bend(landmarks, prev_landmarks=None) -> bool:
    """Breeze — one wrist raised above its shoulder with tilted shoulders."""
    points = _points(
        landmarks,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_WRIST,
        LM.RIGHT_WRIST,
    )
    if points is None:
        return False
    l_sh, r_sh, l_wr, r_wr = points

    shoulder_elevation = abs(l_sh[1] - r_sh[1]) / _width(l_sh, r_sh)

    left_up = l_wr[1] < l_sh[1] - P.SIDE_BEND_WRIST_ABOVE_SHOULDER
    right_up = r_wr[1] < r_sh[1] - P.SIDE_BEND_WRIST_ABOVE_SHOULDER
    if left_up == right_up:
        return False  # need exactly one hand up

    return shoulder_elevation >= P.SIDE_BEND_SHOULDER_ELEVATION


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

    if prev_landmarks is not None:
        prev_points = _points(
            prev_landmarks,
            LM.NOSE,
            LM.LEFT_SHOULDER,
            LM.RIGHT_SHOULDER,
        )
        if prev_points is not None:
            prev_nose, prev_l_sh, prev_r_sh = prev_points
            prev_shoulder_mid = _mid(prev_l_sh, prev_r_sh)
            current_head_height = nose[1] - mid[1]
            previous_head_height = prev_nose[1] - prev_shoulder_mid[1]
            if (
                abs(current_head_height - previous_head_height) / shoulder_w
                >= P.NECK_TURN_NOSE_VERTICAL_DELTA
            ):
                return True

    return False


def detect_bend_down(landmarks, prev_landmarks=None) -> bool:
    """Rain — forward fold / bend down (seated or standing)."""
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

    sh = _mid(l_sh, r_sh)
    hip = _mid(l_hip, r_hip)

    torso_x = sh[0] - hip[0]
    torso_y = hip[1] - sh[1]
    if torso_y < 0.05:
        return False  # hips not below shoulders — bad crop / lying down

    # Measure torso lean from vertical in the image plane.
    bend_angle = math.degrees(math.atan2(abs(torso_x), torso_y))
    return bend_angle + 1e-6 >= P.BEND_DOWN_MIN_ANGLE_DEG


def _max_leg_extension_angle(landmarks) -> float | None:
    """Largest 3D knee angle (hip-knee-ankle) across both legs, in degrees."""
    angles = []
    for hip_i, knee_i, ankle_i in (
        (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ):
        points = _points(landmarks, hip_i, knee_i, ankle_i)
        if points is None:
            continue
        hip, knee, ankle = points
        angles.append(
            _angle_deg(_vec(knee, hip, dims=3), _vec(knee, ankle, dims=3))
        )
    return max(angles) if angles else None

def leg_extension_intensity(landmarks) -> float:
    """Return a 0–1 Wind volume scale based on the most-extended knee."""
    angle = _max_leg_extension_angle(landmarks)
    if angle is None or angle <= P.LEG_EXT_MIN_KNEE_ANGLE + 1e-6:
        return 0.0
    angle_range = P.LEG_EXT_FULL_VOLUME_ANGLE - P.LEG_EXT_MIN_KNEE_ANGLE
    return min(1.0, (angle - P.LEG_EXT_MIN_KNEE_ANGLE) / angle_range)


def detect_leg_extension(landmarks, prev_landmarks=None) -> bool:
    """Wind — one or both legs extended."""
    angle = _max_leg_extension_angle(landmarks)
    return angle is not None and angle > P.LEG_EXT_MIN_KNEE_ANGLE + 1e-6


def torso_shoulder_hip_width_ratio(landmarks) -> float | None:
    """Return shoulder width relative to hip width when both are visible."""
    points = _points(
        landmarks,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if points is None:
        return None
    l_sh, r_sh, l_hip, r_hip = points
    hip_width = _width(l_hip, r_hip)
    if hip_width <= 1e-3:
        return None
    return _width(l_sh, r_sh) / hip_width


def torso_turn_angle(landmarks, neutral_width_ratio: float | None) -> float:
    """Estimate turn angle from shoulder-width reduction relative to the hips."""
    current_ratio = torso_shoulder_hip_width_ratio(landmarks)
    if current_ratio is None or neutral_width_ratio is None or neutral_width_ratio <= 0:
        return 0.0
    width_ratio = min(1.0, current_ratio / neutral_width_ratio)
    return math.degrees(math.acos(width_ratio))


def torso_turn_intensity(landmarks, neutral_width_ratio: float | None) -> float:
    """Return a 0–1 Water volume scale from turn angle."""
    angle = torso_turn_angle(landmarks, neutral_width_ratio)
    if angle <= P.TORSO_TURN_MIN_ANGLE_DEG + 1e-6:
        return 0.0
    angle_range = (
        P.TORSO_TURN_FULL_VOLUME_ANGLE_DEG - P.TORSO_TURN_MIN_ANGLE_DEG
    )
    return min(
        1.0,
        (angle - P.TORSO_TURN_MIN_ANGLE_DEG) / angle_range,
    )


def detect_torso_turn(
    landmarks, prev_landmarks=None, neutral_width_ratio: float | None = None
) -> bool:
    """Water — turn more than 15° from the person's neutral shoulder width."""
    return (
        torso_turn_angle(landmarks, neutral_width_ratio)
        > P.TORSO_TURN_MIN_ANGLE_DEG + 1e-6
    )


def detect_heel_raise(landmarks, prev_landmarks=None) -> bool:
    """Leaves — ankle moving while its leg is extended (seated).

    Per leg, both must hold:
      1. the knee angle is above P.LEG_EXT_MIN_KNEE_ANGLE (leg out), and
      2. the toe moved relative to the ankle by at least
         P.HEEL_RAISE_ANKLE_MOVEMENT (in shin lengths) since the
         previous frame.
    """
    if prev_landmarks is None:
        return False  # movement needs a previous frame

    for hip_i, knee_i, ankle_i, toe_i in (
        (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
    ):
        current = _points(landmarks, hip_i, knee_i, ankle_i, toe_i)
        previous = _points(prev_landmarks, ankle_i, toe_i)
        if current is None or previous is None:
            continue
        hip, knee, ankle, toe = current
        previous_ankle, previous_toe = previous

        # 1. Leg must be extended out.
        knee_angle = _angle_deg(
            _vec(knee, hip, dims=3), _vec(knee, ankle, dims=3)
        )
        if knee_angle <= P.LEG_EXT_MIN_KNEE_ANGLE + 1e-6:
            continue

        # 2. Foot must be moving relative to the ankle.
        foot = _vec(ankle, toe, dims=3)
        previous_foot = _vec(previous_ankle, previous_toe, dims=3)
        shin_length = max(_dist3(knee, ankle), 1e-3)
        movement = (
            math.hypot(*(c - p for c, p in zip(foot, previous_foot)))
            / shin_length
        )
        if movement >= P.HEEL_RAISE_ANKLE_MOVEMENT - 1e-6:
            return True
    return False


# --- Hand detectors --------------------------------------------------


def detect_finger_extension(hand_landmarks: Sequence) -> bool:
    """Detect an open palm with all four fingers extended."""
    wrist = _pt(hand_landmarks, HAND.WRIST)

    def tip_to_pip_ratio(tip: int, pip: int) -> float | None:
        tip_d = _dist2(_pt(hand_landmarks, tip), wrist)
        pip_d = _dist2(_pt(hand_landmarks, pip), wrist)
        if pip_d < 1e-6:
            return None
        return tip_d / pip_d

    ratios = [tip_to_pip_ratio(tip, pip) for tip, pip in FINGER_JOINTS]
    extended_count = sum(
        ratio is not None and ratio >= P.FINGER_EXT_TIP_RATIO
        for ratio in ratios
    )
    return extended_count >= P.FINGER_EXT_MIN_COUNT


def detect_fist(hand_landmarks: Sequence) -> bool:
    """Detect a closed fist from all four fingers being flexed."""
    wrist = _pt(hand_landmarks, HAND.WRIST)
    for tip, pip in FINGER_JOINTS:
        tip_d = _dist2(_pt(hand_landmarks, tip), wrist)
        pip_d = _dist2(_pt(hand_landmarks, pip), wrist)
        if pip_d < 1e-6 or tip_d / pip_d > P.FIST_TIP_RATIO_MAX:
            return False
    return True


# Pose-landmark detectors run by PoseDetector, in priority order.
POSE_DETECTORS: list[tuple[Movement, Callable]] = [
    (Movement.SIDE_BEND, detect_side_bend),
    (Movement.NECK_TURN, detect_neck_turn),
    (Movement.BEND_DOWN, detect_bend_down),
    (Movement.LEG_EXTENSION, detect_leg_extension),
    (Movement.TORSO_TURN, detect_torso_turn),
    (Movement.HEEL_RAISE, detect_heel_raise),
]
