"""
Movement sequences → sound.

A sequence completes when one person starts each of its steps in order
within `within_seconds`. Other movements done in between are ignored.
Its sound plays once, and is skipped if that sound is already playing.

Every sound used here must be in ALL_SOUNDS in movements/mapping.py,
and the longest window must fit in HISTORY_SECONDS in parameters.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from movements.mapping import LEAVES, Movement, Sound


@dataclass(frozen=True)
class MovementSequence:
    name: str  # shown in the UI
    steps: tuple[Movement, ...]
    within_seconds: float
    sound: Sound


SEQUENCES: list[MovementSequence] = [
    MovementSequence(
        name="Side bend → bend down",
        steps=(Movement.SIDE_BEND, Movement.BEND_DOWN),
        within_seconds=5.0,
        sound=LEAVES,
    ),
]


def is_completed(
    sequence: MovementSequence,
    starts: Sequence[tuple[float, Movement]],
    after: float,
) -> bool:
    """
    True if the newest entry in `starts` completes `sequence`.

    `starts` is one person's (time, movement) history of movement
    starts, oldest first. Every step must have started inside the time
    window and strictly after `after` (when this sequence last fired),
    so one run of movements can't complete the same sequence twice.
    """
    if not starts or starts[-1][1] != sequence.steps[-1]:
        return False

    window_start = starts[-1][0] - sequence.within_seconds
    remaining = list(sequence.steps)

    # Walk back from the newest start, ticking off steps last to first.
    for time, movement in reversed(starts):
        if time < window_start or time <= after:
            return False
        if movement == remaining[-1]:
            remaining.pop()
            if not remaining:
                return True

    return False
