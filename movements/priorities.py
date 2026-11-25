"""
Movement priorities: when one movement is happening, which others are
ignored.

Each key suppresses the movements in its set. Keep the table acyclic
(A suppresses B, B never suppresses A). Applied in
detection/person_tracker.py after the enabled/disabled filter.
"""

from collections.abc import Set

from movements.mapping import Movement

SUPPRESSES: dict[Movement, frozenset[Movement]] = {
    # Side bend tilts the shoulders and head; don't also fire twist/neck.
    Movement.SIDE_BEND: frozenset({Movement.NECK_TURN, Movement.TORSO_TURN}),
    # A forward fold shifts the head and shortens the torso, which looks
    # like a neck turn or torso turn.
    Movement.BEND_DOWN: frozenset({Movement.NECK_TURN, Movement.TORSO_TURN}),
    # Prevent torso turn from also doing any neck turn shenanigans
    Movement.TORSO_TURN: frozenset({Movement.NECK_TURN}),
    # If doing leg stuff, ignorw upper body things
    Movement.LEG_MOVEMENT: frozenset({Movement.TORSO_TURN, Movement.SIDE_BEND, Movement.BEND_DOWN}),
}


def apply_priorities(
    detected: Set[Movement], active: Set[Movement] = frozenset()
) -> set[Movement]:
    """Drop detections that a higher-priority movement overrides.

    A movement dominates if it is detected this frame OR still active
    (already smoothed by the tracker), so a brief detection flicker in
    the dominant movement doesn't let a suppressed one through.
    """
    dominant = set(detected) | set(active)
    suppressed: set[Movement] = set()
    for movement in dominant:
        suppressed |= SUPPRESSES.get(movement, frozenset())
    return set(detected) - suppressed