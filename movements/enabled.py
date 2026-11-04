"""
Turn movements and sequences on or off — handy for testing a few at a time.

A disabled movement never shows on screen, never plays a sound, and can't
count as a sequence step. This is enforced in detection/person_tracker.py.
If every pose movement is off the pose model is skipped, and if both hand
movements are off the hand model is skipped, so testing gets faster too.

Anything missing from these dicts counts as enabled; main.py warns about
it at startup.
"""

from movements.mapping import Movement
from movements.sequences import (
    FIST_OPEN_PALM_FIST,
    SEQUENCES,
    SIDE_BEND_THEN_BEND_DOWN,
    MovementSequence,
)

ENABLED_MOVEMENTS: dict[Movement, bool] = {
    Movement.SIDE_BEND: False,
    Movement.NECK_TURN: False, # If move neck up and down too slowly, it will not detect since the position difference between frames is smaller than the threshold but still moving. 
    # We think it works good enough to keep since less people will move that slow.
    Movement.BEND_DOWN: False,
    Movement.LEG_EXTENSION: True,
    Movement.TORSO_TURN: False,
    Movement.HEEL_RAISE: True, #Changed a LOT of stuff for this, still needs a lot of testing and stuff to tune parameters and calculations
    Movement.FINGER_EXTENSION: False,
    Movement.FIST: False,
}

ENABLED_SEQUENCES: dict[MovementSequence, bool] = {
    SIDE_BEND_THEN_BEND_DOWN: True,
    FIST_OPEN_PALM_FIST: True,
}


def is_enabled(movement: Movement) -> bool:
    return ENABLED_MOVEMENTS.get(movement, True)


def sequence_enabled(sequence: MovementSequence) -> bool:
    return ENABLED_SEQUENCES.get(sequence, True)


def summary() -> str:
    """What's switched on, plus sequences that can't fire as configured."""
    on = [m.value for m in Movement if is_enabled(m)]
    off = [m.value for m in Movement if not is_enabled(m)]
    lines = [f"Movements on: {', '.join(on) or 'none'}"]
    if off:
        lines.append(f"Movements off: {', '.join(off)}")

    unlisted = [m.value for m in Movement if m not in ENABLED_MOVEMENTS]
    unlisted += [s.name for s in SEQUENCES if s not in ENABLED_SEQUENCES]
    if unlisted:
        lines.append(
            f"Not listed in movements/enabled.py (treated as on): "
            f"{', '.join(unlisted)}"
        )

    for sequence in SEQUENCES:
        if not sequence_enabled(sequence):
            lines.append(f"Sequence off: {sequence.name}")
            continue
        blocked = [m.value for m in sequence.steps if not is_enabled(m)]
        if blocked:
            lines.append(
                f"Sequence '{sequence.name}' can't fire: "
                f"{', '.join(blocked)} is off"
            )
    return "\n".join(lines)
