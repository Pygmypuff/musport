"""
Turn movements and sequences on or off — handy for testing a few at a time.

A disabled movement isn't detected at all: its rule doesn't run, it never
shows on screen or plays a sound, and it can't count as a sequence step.
If every pose movement is off the pose model is skipped, and if finger
extension is off the hand model is skipped, so testing gets faster too.

Anything missing from these dicts counts as enabled.
"""

from movements.mapping import Movement
from movements.sequences import SEQUENCES, SIDE_BEND_THEN_BEND_DOWN, MovementSequence

ENABLED_MOVEMENTS: dict[Movement, bool] = {
    Movement.SIDE_BEND: True,
    Movement.NECK_TURN: True,
    Movement.BEND_DOWN: True,
    Movement.LEG_EXTENSION: True,
    Movement.TORSO_TURN: True,
    Movement.HEEL_RAISE: True,
    Movement.FINGER_EXTENSION: True,
}

ENABLED_SEQUENCES: dict[MovementSequence, bool] = {
    SIDE_BEND_THEN_BEND_DOWN: True,
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
