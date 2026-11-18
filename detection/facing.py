"""
Which way each person is sitting, and who controls the sounds.

People sit in a semicircle, so some are turned partly or fully sideways
to the camera, where pose landmarks are unreliable. Everyone is meant to
be doing the same movement, so one person is enough: the one sitting
straightest to the camera controls the sounds (ControllerPicker).

A person's resting direction is the median of their shoulder direction
over the last RESTING_DIRECTION_SECONDS. A torso twist lasting a few
seconds barely moves it; the way their chair faces does.

Thresholds live in movements/parameters.py.
"""

from __future__ import annotations

import statistics
from collections import deque

from movements import parameters as P


def _wrap180(angle: float) -> float:
    """Wrap an angle difference to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


class PersonFacing:
    """One person's resting direction."""

    def __init__(self):
        # (time, shoulder direction in degrees; 0 = facing the camera)
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, yaw: float, now: float):
        """Record this frame's shoulder direction."""
        self._samples.append((now, yaw))
        while now - self._samples[0][0] > P.RESTING_DIRECTION_SECONDS:
            self._samples.popleft()

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


class ControllerPicker:
    """Chooses the one person whose movements control the sounds.

    That's whoever's resting direction is closest to facing the camera.
    Control only moves to someone else when they're straighter by at
    least CONTROLLER_SWITCH_MARGIN_DEG, or when the controller leaves, so
    two people at similar angles don't keep swapping it.
    """

    def __init__(self):
        self.current: int | None = None

    def update(self, angles: dict[int, float | None]) -> int | None:
        """`angles`: {person_id: angle from camera, or None if not known yet}
        for everyone being tracked. Returns the controller's ID, or None if
        nobody's direction is known yet."""
        known = {pid: angle for pid, angle in angles.items() if angle is not None}
        if not known:
            self.current = None
            return None

        straightest = min(known, key=known.get)
        if (
            self.current not in known
            or known[straightest]
            < known[self.current] - P.CONTROLLER_SWITCH_MARGIN_DEG
        ):
            self.current = straightest
        return self.current
