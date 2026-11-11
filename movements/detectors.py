"""
Pose / hand landmark → yoga Movement IDs.

Each detector is a geometric rule tuned for seated senior classes.
Thresholds live in movements/parameters.py so they can be adjusted
without rewriting the math.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from mediapipe.tasks.python.vision import PoseLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from movements import parameters as P
from movements.mapping import Movement

LM = PoseLandmark
HAND = HandLandmark

# (mcp, pip, tip) joints of the four non-thumb fingers.
FINGER_JOINTS = (
    (HAND.INDEX_FINGER_MCP, HAND.INDEX_FINGER_PIP, HAND.INDEX_FINGER_TIP),
    (HAND.MIDDLE_FINGER_MCP, HAND.MIDDLE_FINGER_PIP, HAND.MIDDLE_FINGER_TIP),
    (HAND.RING_FINGER_MCP, HAND.RING_FINGER_PIP, HAND.RING_FINGER_TIP),
    (HAND.PINKY_MCP, HAND.PINKY_PIP, HAND.PINKY_TIP),
)

# --- Parameters ------------------------------------------------------


def _p(name: str, default=None):
    """Read a parameter from movements.parameters, falling back to a default."""
    if hasattr(P, name):
        return getattr(P, name)
    if default is not None:
        return default
    raise AttributeError(f"Unknown movement parameter: {name}")


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


def _geo(landmarks, world, *indices):
    """Like _points, but coordinates come from `world` when available.

    Presence is always checked on the normalised `landmarks` (they carry the
    reliable confidence values); geometry is read from the metric world
    landmarks, which give undistorted angles.
    """
    if not _present(landmarks, *indices):
        return None
    src = world if world is not None else landmarks
    return [_pt(src, i) for i in indices]


# --- Geometry --------------------------------------------------------


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _dist2(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3(a, b) -> float:
    return math.dist(a[:3], b[:3])


def _width(a, b) -> float:
    """Image-plane distance between two points, floored so it is safe to divide by."""
    return max(_dist2(a, b), 1e-3)


def _vec(a, b, dims: int = 2) -> tuple[float, ...]:
    """Vector from a to b using the first `dims` coordinates."""
    return tuple(b[i] - a[i] for i in range(dims))


def _angle_deg(u, v, as_lines: bool = False) -> float:
    """Angle between vectors u and v in degrees [0, 180] (2D or 3D)."""
    nu = math.hypot(*u)
    nv = math.hypot(*v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_a = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    if as_lines:
        cos_a = abs(cos_a)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear in between."""
    if hi <= lo:
        return 1.0 if x > lo else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _yaw_deg(left, right) -> float:
    """Yaw of the line left→right in the x/z plane (best with world landmarks)."""
    return math.degrees(math.atan2(right[2] - left[2], right[0] - left[0]))


def _wrap180(angle: float) -> float:
    """Wrap an angle difference to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def _knee_angle(hip, knee, ankle) -> float:
    """3D knee angle (hip-knee-ankle) in degrees; 180 = straight leg."""
    return _angle_deg(_vec(knee, hip, dims=3), _vec(knee, ankle, dims=3))


def _torso_lean_deg(l_sh, r_sh, l_hip, r_hip) -> tuple[float, float]:
    """(forward_deg, lateral_deg): torso lean split into its two components.

    Uses the hip-mid → shoulder-mid vector. Image/world y points down, so
    "up" along the torso is hip_y - shoulder_y. Angles > 90° mean the
    shoulders are below the hips (deep fold).
    """
    sh = _mid(l_sh, r_sh)
    hip = _mid(l_hip, r_hip)
    up = hip[1] - sh[1]
    forward = math.degrees(math.atan2(abs(sh[2] - hip[2]), up))
    lateral = math.degrees(math.atan2(abs(sh[0] - hip[0]), up))
    return forward, lateral


# --- Landmark smoothing (One Euro filter) ----------------------------


class _OneEuro:
    """One Euro filter for a single scalar: low jitter at rest, low lag in motion."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t: float | None = None
        self._x = 0.0
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx += a_d * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x += a * (x - self._x)
        self._t = t
        return self._x


@dataclass
class SmoothedLandmark:
    x: float
    y: float
    z: float
    visibility: float = 0.0
    presence: float | None = None


class LandmarkSmoother:
    """Smooths a list of landmarks over time (x/y/z only; confidences pass through)."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 3.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[int, tuple[_OneEuro, _OneEuro, _OneEuro]] = {}

    def reset(self) -> None:
        self._filters.clear()

    def __call__(self, landmarks, t: float) -> list[SmoothedLandmark]:
        out = []
        for i, lm in enumerate(landmarks):
            f = self._filters.get(i)
            if f is None:
                f = tuple(_OneEuro(self.min_cutoff, self.beta) for _ in range(3))
                self._filters[i] = f
            out.append(
                SmoothedLandmark(
                    f[0](float(lm.x), t),
                    f[1](float(lm.y), t),
                    f[2](float(lm.z), t),
                    float(getattr(lm, "visibility", 0.0) or 0.0),
                    getattr(lm, "presence", None),
                )
            )
        return out


# --- Pose scores (each returns 0–1) ----------------------------------


def side_bend_score(landmarks, world=None) -> float:
    """Breeze — torso leaning sideways, ideally with one arm raised."""
    pts = _geo(
        landmarks,
        world,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_HIP,
        LM.RIGHT_HIP,
    )
    if pts is None:
        return 0.0
    _, lateral = _torso_lean_deg(*pts)
    score = _ramp(
        lateral,
        _p("SIDE_BEND_MIN_LATERAL_DEG"),
        _p("SIDE_BEND_FULL_LATERAL_DEG"),
    )

    # Exactly one wrist above its shoulder (image plane) confirms the pose.
    one_arm_up = False
    arms = _points(
        landmarks,
        LM.LEFT_SHOULDER,
        LM.RIGHT_SHOULDER,
        LM.LEFT_WRIST,
        LM.RIGHT_WRIST,
    )
    if arms is not None:
        l_sh, r_sh, l_wr, r_wr = arms
        left_up = l_wr[1] < l_sh[1] - P.SIDE_BEND_WRIST_ABOVE_SHOULDER
        right_up = r_wr[1] < r_sh[1] - P.SIDE_BEND_WRIST_ABOVE_SHOULDER
        one_arm_up = left_up != right_up

    return score * (1.0 if one_arm_up else _p("SIDE_BEND_NO_ARM_FACTOR"))


def bend_down_score(landmarks, world=None) -> float:
    """Rain — forward fold.

    Needs BOTH a forward torso lean (signed, shoulders toward the camera)
    AND the head dropping toward shoulder level. Either alone is too noisy.
    """
    pts = _geo(
        landmarks, world,
        LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP,
    )
    if pts is None:
        return 0.0
    l_sh, r_sh, l_hip, r_hip = pts
    sh, hip = _mid(l_sh, r_sh), _mid(l_hip, r_hip)

    # Cue 1: signed forward lean. MediaPipe z is smaller toward the camera,
    # so shoulders closer than hips => positive. Leaning back scores 0.
    up = hip[1] - sh[1]
    forward = math.degrees(math.atan2(hip[2] - sh[2], up))
    lo = P.BEND_DOWN_MIN_ANGLE_DEG
    hi = _p("BEND_DOWN_FULL_ANGLE_DEG", lo + 25.0)
    lean = _ramp(forward, lo, hi)

    # Cue 2: nose height relative to the shoulder line, in shoulder widths
    # (image plane). Upright is clearly negative (nose above shoulders).
    head = _points(landmarks, LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if head is None:
        return 0.0  # can't confirm the fold, so don't claim it
    nose, i_l_sh, i_r_sh = head
    drop = (nose[1] - _mid(i_l_sh, i_r_sh)[1]) / _width(i_l_sh, i_r_sh)
    head_score = _ramp(
        drop,
        _p("BEND_DOWN_HEAD_DROP_MIN", -0.25),
        _p("BEND_DOWN_HEAD_DROP_FULL", 0.15),
    )

    return min(lean, head_score)


def shoulder_yaw_deg(landmarks, world) -> float | None:
    """Which way the shoulders face, in degrees.

    0 = square to the camera, ±90 = side-on (the sign says which side),
    ±180 = back to the camera. Read from MediaPipe's 3D world landmarks
    (metres), which have real depth; the 2D landmarks' depth is too rough
    for this. Hips aren't used: seated, they're hidden by the thighs and
    chair. None if the shoulders aren't visible.
    """
    if world is None or not _present(landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER):
        return None
    left = _pt(world, LM.LEFT_SHOULDER)
    right = _pt(world, LM.RIGHT_SHOULDER)
    return math.degrees(math.atan2(left[2] - right[2], left[0] - right[0]))


def torso_turn_score(yaw: float | None, resting_yaw: float | None) -> float:
    """Water — shoulders turned away from the person's resting direction.

    Measured against where this person normally faces (detection/facing.py),
    so someone whose chair is angled to the camera isn't always "turning".
    0 below TORSO_TURN_MIN_ANGLE_DEG, 1 at the full-volume angle.
    """
    if yaw is None or resting_yaw is None:
        return 0.0
    twist = abs(_wrap180(yaw - resting_yaw))
    return _ramp(
        twist, P.TORSO_TURN_MIN_ANGLE_DEG, P.TORSO_TURN_FULL_VOLUME_ANGLE_DEG
    )


def neck_turn_score(landmarks, world=None, prev_landmarks=None) -> float:
    """Birds — head yaw RELATIVE to the shoulders.

    Because it is relative, a pure torso rotation (head and shoulders turn
    together) scores ~0. Nose offset also catches turns in either direction,
    and relative nose-height change catches a downward head tilt.
    """
    score = 0.0
    pts = _geo(
        landmarks, world,
        LM.LEFT_EAR, LM.RIGHT_EAR, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    )
    if pts is not None:
        l_ear, r_ear, l_sh, r_sh = pts
        rel = abs(_wrap180(_yaw_deg(l_ear, r_ear) - _yaw_deg(l_sh, r_sh)))
        score = _ramp(rel, _p("NECK_TURN_MIN_DEG"), _p("NECK_TURN_FULL_DEG"))

    # Absolute offset treats left and right turns symmetrically.
    pts = _points(landmarks, LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    if pts is not None:
        nose, l_sh, r_sh = pts
        offset = abs(nose[0] - _mid(l_sh, r_sh)[0]) / _width(l_sh, r_sh)
        lo = P.NECK_TURN_NOSE_OFFSET
        hi = _p("NECK_TURN_NOSE_OFFSET_FULL", lo * 2.0)
        score = max(score, _ramp(offset, lo, hi))

    if prev_landmarks is not None:
        current = _points(
            landmarks, LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER
        )
        previous = _points(
            prev_landmarks, LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER
        )
        if current is not None and previous is not None:
            nose, l_sh, r_sh = current
            prev_nose, prev_l_sh, prev_r_sh = previous
            current_height = nose[1] - _mid(l_sh, r_sh)[1]
            previous_height = prev_nose[1] - _mid(prev_l_sh, prev_r_sh)[1]
            downward_delta = (current_height - previous_height) / _width(l_sh, r_sh)
            lo = P.NECK_TURN_NOSE_VERTICAL_DELTA
            score = max(score, _ramp(downward_delta, lo, lo * 2.0))
    return score


def _max_leg_extension_angle(landmarks, world=None) -> float | None:
    """Largest 3D knee angle (hip-knee-ankle) across both legs, in degrees."""
    angles = []
    for hip_i, knee_i, ankle_i in (
        (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ):
        pts = _geo(landmarks, world, hip_i, knee_i, ankle_i)
        if pts is None:
            continue
        angles.append(_knee_angle(*pts))
    return max(angles) if angles else None


def leg_extension_intensity(landmarks, world=None) -> float:
    """0–1 Wind volume scale based on the most-extended knee."""
    angle = _max_leg_extension_angle(landmarks, world)
    if angle is None:
        return 0.0
    return _ramp(angle, P.LEG_EXT_MIN_KNEE_ANGLE, P.LEG_EXT_FULL_VOLUME_ANGLE)


leg_extension_score = leg_extension_intensity


# --- Stateful scorer + heel raise ------------------------------------


class PoseScorer:
    """Smooths landmarks and produces a 0–1 score per pose Movement.

    Stateful because (a) landmark smoothing needs history and (b) heel
    raise measures how much the ankle angle changes over a short window,
    which is independent of frame rate (unlike a frame-to-frame delta).
    """

    def __init__(self, smooth: bool = True):
        self._pose_filter = LandmarkSmoother() if smooth else None
        self._world_filter = LandmarkSmoother() if smooth else None
        # side -> deque[(t, ankle_flexion_deg)]
        self._ankle: dict[str, deque] = {"L": deque(), "R": deque()}

    def reset(self) -> None:
        if self._pose_filter:
            self._pose_filter.reset()
            self._world_filter.reset()
        for q in self._ankle.values():
            q.clear()

    def _heel_raise_score(self, landmarks, world, t: float) -> float:
        window = _p("HEEL_RAISE_WINDOW_S")
        best = 0.0
        legs = (
            ("L", LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
            ("R", LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
        )
        for side, hip_i, knee_i, ankle_i, toe_i in legs:
            q = self._ankle[side]
            pts = _geo(landmarks, world, hip_i, knee_i, ankle_i, toe_i)
            # Leg must be extended out, as before; otherwise forget this side.
            if pts is None or _knee_angle(*pts[:3]) <= P.LEG_EXT_MIN_KNEE_ANGLE:
                q.clear()
                continue
            hip, knee, ankle, toe = pts
            flexion = _angle_deg(
                _vec(ankle, knee, dims=3), _vec(ankle, toe, dims=3)
            )
            q.append((t, flexion))
            while q and t - q[0][0] > window:
                q.popleft()
            angles = [a for _, a in q]
            rng = max(angles) - min(angles)
            best = max(
                best,
                _ramp(rng, _p("HEEL_RAISE_MIN_RANGE_DEG"), _p("HEEL_RAISE_FULL_RANGE_DEG")),
            )
        return best

    def score(self, landmarks, world=None, t: float | None = None) -> dict:
        """Return {Movement: 0–1}. `landmarks=None` (no person) resets state."""
        t = time.monotonic() if t is None else t
        if landmarks is None:
            self.reset()
            return {}
        if self._pose_filter is not None:
            landmarks = self._pose_filter(landmarks, t)
            if world is not None:
                world = self._world_filter(world, t)
        return {
            Movement.BEND_DOWN: bend_down_score(landmarks, world),
            Movement.SIDE_BEND: side_bend_score(landmarks, world),
            Movement.TORSO_TURN: torso_turn_score(
                shoulder_yaw_deg(landmarks, world), resting_yaw=0.0
            ),
            Movement.NECK_TURN: neck_turn_score(landmarks, world),
            Movement.HEEL_RAISE: self._heel_raise_score(landmarks, world, t),
            Movement.LEG_EXTENSION: leg_extension_score(landmarks, world),
        }


# --- Arbiter: smoothing, debounce, hierarchy -------------------------

# Highest priority first.
RANKING: list[Movement] = [
    Movement.BEND_DOWN,
    Movement.SIDE_BEND,
    Movement.TORSO_TURN,
    Movement.NECK_TURN,
    Movement.HEEL_RAISE,
    Movement.LEG_EXTENSION,
]

# A movement on the left silences the movements on the right. A bigger
# movement mechanically drags smaller ones along (turning the torso carries
# the head), so the smaller ones are treated as side effects. Heel raise and
# leg extension are deliberately NOT suppressed: they layer with the rest.
SUPPRESSES: dict[Movement, set[Movement]] = {
    Movement.BEND_DOWN: {Movement.SIDE_BEND, Movement.TORSO_TURN, Movement.NECK_TURN},
    Movement.SIDE_BEND: {Movement.TORSO_TURN, Movement.NECK_TURN},
    Movement.TORSO_TURN: {Movement.NECK_TURN},
}


@dataclass
class _State:
    smoothed: float = 0.0
    active: bool = False
    above_since: float | None = None
    below_since: float | None = None
    last_active: float = field(default=float("-inf"))


class MovementArbiter:
    """Turns raw 0–1 scores into a stable set of active movements.

    Scores ramp from "movement threshold" (0) to "full movement" (1), so the
    default on/off levels are low: `on=0.25` means a quarter of the way up the
    ramp. All timings are in seconds, so behaviour doesn't depend on FPS.

    tau_s      score smoothing time constant (bigger = smoother, slower)
    on / off   hysteresis: must reach `on` to start, fall below `off` to stop
    min_on_s   score must stay >= on this long before the movement starts
    min_off_s  score must stay < off this long before the movement stops
    hold_s     after a movement ends it keeps suppressing lower-ranked ones
               this long (stops e.g. Birds popping in as a torso turn ends)
    """

    def __init__(
        self,
        tau_s: float = 0.15,
        on: float = 0.25,
        off: float = 0.10,
        min_on_s: float = 0.12,
        min_off_s: float = 0.30,
        hold_s: float = 0.40,
        ranking: Sequence[Movement] = RANKING,
        suppresses: Mapping[Movement, set[Movement]] = SUPPRESSES,
    ):
        self.tau_s, self.on, self.off = tau_s, on, off
        self.min_on_s, self.min_off_s, self.hold_s = min_on_s, min_off_s, hold_s
        self.ranking = list(ranking)
        self.suppresses = suppresses
        self.state: dict[Movement, _State] = {m: _State() for m in self.ranking}
        self._last_t: float | None = None

    def update(self, raw: Mapping[Movement, float], now: float) -> dict[Movement, float]:
        """raw: {Movement: score 0–1}. Returns {Movement: intensity 0–1} to play."""
        dt = 1 / 30 if self._last_t is None else max(now - self._last_t, 1e-3)
        self._last_t = now
        alpha = 1.0 - math.exp(-dt / self.tau_s)

        for m, st in self.state.items():
            st.smoothed += alpha * (raw.get(m, 0.0) - st.smoothed)

            if st.smoothed >= self.on:
                if st.above_since is None:
                    st.above_since = now
                st.below_since = None
            elif st.smoothed < self.off:
                if st.below_since is None:
                    st.below_since = now
                st.above_since = None
            else:  # inside the hysteresis band: neither timer runs
                st.above_since = st.below_since = None

            if (
                not st.active
                and st.above_since is not None
                and now - st.above_since >= self.min_on_s
            ):
                st.active = True
            elif (
                st.active
                and st.below_since is not None
                and now - st.below_since >= self.min_off_s
            ):
                st.active = False
            if st.active:
                st.last_active = now

        blocked: set[Movement] = set()
        out: dict[Movement, float] = {}
        for m in self.ranking:
            if m in blocked:
                continue
            st = self.state[m]
            # Claim as soon as the score is high (even before debounce says
            # "active") and keep claiming briefly after release.
            claiming = (
                st.active
                or st.smoothed >= self.on
                or now - st.last_active < self.hold_s
            )
            if claiming:
                blocked |= self.suppresses.get(m, set())
            if st.active:
                out[m] = min(1.0, st.smoothed)
        return out


class MovementPipeline:
    """PoseScorer + MovementArbiter. One call per video frame."""

    def __init__(
        self,
        scorer: PoseScorer | None = None,
        arbiter: MovementArbiter | None = None,
    ):
        self.scorer = scorer or PoseScorer()
        self.arbiter = arbiter or MovementArbiter()
        self.last_scores: dict = {}  # raw scores of the latest frame, for logging

    def update(self, landmarks, world=None, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        self.last_scores = self.scorer.score(landmarks, world, now)
        return self.arbiter.update(self.last_scores, now)


# --- Hand detectors --------------------------------------------------


def _pip_angles(hand_landmarks: Sequence) -> list[float]:
    """3D angle at the PIP joint (MCP-PIP-TIP) for each non-thumb finger.

    180° = straight finger, small = curled. Being an angle (not a distance
    to the wrist) it still works when the hand points at the camera. Pass
    hand *world* landmarks if you have them for the most accurate result.
    """
    angles = []
    for mcp, pip, tip in FINGER_JOINTS:
        a, b, c = (_pt(hand_landmarks, i) for i in (mcp, pip, tip))
        angles.append(_angle_deg(_vec(b, a, dims=3), _vec(b, c, dims=3)))
    return angles


def detect_finger_extension(hand_landmarks: Sequence) -> bool:
    """Open palm with enough fingers extended."""
    lo = _p("FINGER_EXT_MIN_PIP_ANGLE_DEG")
    extended = sum(angle >= lo for angle in _pip_angles(hand_landmarks))
    return extended >= P.FINGER_EXT_MIN_COUNT


def detect_fist(hand_landmarks: Sequence) -> bool:
    """Closed fist: all four fingers clearly flexed."""
    hi = _p("FIST_MAX_PIP_ANGLE_DEG")
    return all(angle <= hi for angle in _pip_angles(hand_landmarks))


# --- Legacy boolean API (no smoothing, no hierarchy) -----------------
# Kept so existing callers keep working. "True" means the score is past the
# movement's minimum threshold (score > 0).


def detect_side_bend(landmarks, prev_landmarks=None, world=None) -> bool:
    return side_bend_score(landmarks, world) > 0.0


def detect_neck_turn(landmarks, prev_landmarks=None, world=None) -> bool:
    return neck_turn_score(landmarks, world, prev_landmarks) > 0.0


def detect_bend_down(landmarks, prev_landmarks=None, world=None) -> bool:
    return bend_down_score(landmarks, world) > 0.0


def detect_leg_extension(landmarks, prev_landmarks=None, world=None) -> bool:
    return leg_extension_score(landmarks, world) > 0.0


def detect_torso_turn(
    landmarks, prev_landmarks=None, world=None, resting_yaw=None
) -> bool:
    return torso_turn_score(shoulder_yaw_deg(landmarks, world), resting_yaw) > 0.0


def detect_heel_raise(landmarks, prev_landmarks=None) -> bool:
    """Leaves — ankle moving while its leg is extended (frame-to-frame version).

    Superseded by PoseScorer's windowed ankle-flexion score, which doesn't
    depend on frame rate. Kept only for the legacy API.
    """
    if prev_landmarks is None:
        return False
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
        if _knee_angle(hip, knee, ankle) <= P.LEG_EXT_MIN_KNEE_ANGLE + 1e-6:
            continue
        foot = _vec(ankle, toe, dims=3)
        previous_foot = _vec(previous_ankle, previous_toe, dims=3)
        shin_length = max(_dist3(knee, ankle), 1e-3)
        movement = (
            math.hypot(*(c - p for c, p in zip(foot, previous_foot))) / shin_length
        )
        if movement >= P.HEEL_RAISE_ANKLE_MOVEMENT - 1e-6:
            return True
    return False


# Legacy pose detectors, in priority order. Prefer MovementPipeline.
POSE_DETECTORS: list[tuple[Movement, Callable]] = [
    (Movement.BEND_DOWN, detect_bend_down),
    (Movement.SIDE_BEND, detect_side_bend),
    (Movement.TORSO_TURN, detect_torso_turn),
    (Movement.NECK_TURN, detect_neck_turn),
    (Movement.HEEL_RAISE, detect_heel_raise),
    (Movement.LEG_EXTENSION, detect_leg_extension),
]
