"""
Which way each person is sitting, and whether they face the camera enough
to be analysed.

People sit in a semicircle, so some are turned partly or fully sideways
to the camera, where pose landmarks are unreliable. Everyone is meant to
be doing the same movement, so it's fine to only analyse the people who
face the camera.

A person's resting direction is the median of their shoulder direction
over the last RESTING_DIRECTION_SECONDS. A torso twist lasting a few
seconds barely moves it; the way their chair faces does. People whose
resting direction is too far from the camera are skipped by the gesture
pipeline, and re-measured every FACING_RECHECK_SECONDS in case they move.

Thresholds live in movements/parameters.py.
"""

import math
import statistics
from collections import deque

from movements import parameters as P


def _wrap180(angle: float) -> float:
    """Wrap an angle difference to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


class PersonFacing:
    """One person's resting direction and whether they're turned away."""

    def __init__(self):
        # (time, shoulder direction in degrees; 0 = facing the camera)
        self._samples: deque[tuple[float, float]] = deque()
        self.turned_away = False
        self._next_check = -math.inf

    def add(self, yaw: float, now: float):
        """Record this frame's shoulder direction and update the decision."""
        self._samples.append((now, yaw))
        while now - self._samples[0][0] > P.RESTING_DIRECTION_SECONDS:
            self._samples.popleft()

        angle = self.angle_from_camera
        if angle is None:
            return
        # Hysteresis so someone right at the limit doesn't flicker in and out.
        if angle >= P.FACING_MAX_TURN_DEG:
            self.turned_away = True
        elif angle <= P.FACING_MAX_TURN_DEG - P.FACING_HYSTERESIS_DEG:
            self.turned_away = False

    @property
    def resting_yaw(self) -> float | None:
        """Median shoulder direction, or None until there are enough samples."""
        if len(self._samples) < P.FACING_MIN_SAMPLES:
            return None
        # Take the median relative to the newest sample, so directions
        # either side of ±180° (facing away) don't average out to 0.
        ref = self._samples[-1][1]
        offset = statistics.median(_wrap180(y - ref) for _, y in self._samples)
        return _wrap180(ref + offset)

    @property
    def angle_from_camera(self) -> float | None:
        """How far the resting direction is from facing the camera (0–180°)."""
        resting = self.resting_yaw
        return None if resting is None else abs(resting)

    def should_skip(self, now: float) -> bool:
        """True while turned away, except when it's time to re-measure."""
        return self.turned_away and now < self._next_check

    def skipped(self, now: float):
        """Call after deciding to skip someone; schedules their next re-check."""
        self._next_check = now + P.FACING_RECHECK_SECONDS
