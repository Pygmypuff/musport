"""
Per-person movement tracking over time.

The detectors report what each person is doing in a single frame, which
flickers. PersonTracker smooths that into start / end events for each
person, keeps a short history of what they did, and reports when they
complete a movement sequence (movements/sequences.py).

Timing thresholds live in movements/parameters.py.
"""

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from movements import parameters as P
from movements.mapping import Movement
from movements.sequences import SEQUENCES, MovementSequence, is_completed


@dataclass(frozen=True)
class MovementEvent:
    person_id: int
    movement: Movement
    kind: Literal["start", "end"]
    time: float


@dataclass(frozen=True)
class SequenceEvent:
    person_id: int
    sequence: MovementSequence
    time: float


Event = MovementEvent | SequenceEvent


class PersonState:
    """What one tracked person is doing now and has done recently."""

    def __init__(self, person_id: int, now: float):
        self.id = person_id
        self.last_seen = now
        # Movements in progress, after smoothing.
        self.active: set[Movement] = set()
        # (time, movement) for each movement start, oldest first.
        self.starts: deque[tuple[float, Movement]] = deque()
        self.last_sequence: SequenceEvent | None = None

        # When the current unbroken run of detections began, per movement.
        self._run_start: dict[Movement, float] = {}
        self._last_detected: dict[Movement, float] = {}
        self._sequence_fired: dict[MovementSequence, float] = {}

    def update(self, detected: set[Movement], now: float) -> list[Event]:
        """Feed this frame's raw detections; return the resulting events."""
        events: list[Event] = []

        # Enum order keeps same-frame starts in a stable order.
        for movement in Movement:
            if movement not in detected:
                continue

            last = self._last_detected.get(movement, -math.inf)
            # A gap shorter than the end delay doesn't break the run.
            if now - last >= P.MOVEMENT_END_SECONDS:
                self._run_start[movement] = now
            self._last_detected[movement] = now

            held_for = now - self._run_start[movement]
            if movement not in self.active and held_for >= P.MOVEMENT_START_SECONDS:
                self.active.add(movement)
                self.starts.append((now, movement))
                events.append(MovementEvent(self.id, movement, "start", now))
                events += self._completed_sequences(now)

        for movement in list(self.active):
            if now - self._last_detected[movement] >= P.MOVEMENT_END_SECONDS:
                self.active.discard(movement)
                events.append(MovementEvent(self.id, movement, "end", now))

        while self.starts and now - self.starts[0][0] > P.HISTORY_SECONDS:
            self.starts.popleft()

        return events

    def _completed_sequences(self, now: float) -> list[SequenceEvent]:
        events = []
        for sequence in SEQUENCES:
            after = self._sequence_fired.get(sequence, -math.inf)
            if is_completed(sequence, self.starts, after):
                self._sequence_fired[sequence] = now
                self.last_sequence = SequenceEvent(self.id, sequence, now)
                events.append(self.last_sequence)
        return events

    def movements_in_progress(self) -> list[Movement]:
        """Active movements in a stable display order."""
        return [m for m in Movement if m in self.active]

    def recent_sequence(self, now: float, seconds: float = 3.0) -> str | None:
        """Name of the sequence completed in the last `seconds`, if any."""
        if self.last_sequence and now - self.last_sequence.time <= seconds:
            return self.last_sequence.sequence.name
        return None


class PersonTracker:
    """Keeps a PersonState per tracked ID and turns detections into events."""

    def __init__(self, on_person_removed: Callable[[int], None] | None = None):
        self.people: dict[int, PersonState] = {}
        self._on_person_removed = on_person_removed

    def update(
        self, detections: dict[int, list[Movement]], now: float
    ) -> list[Event]:
        """
        Feed one frame of {person_id: movements detected}. People missing
        from `detections` are treated as doing nothing, and are forgotten
        once they've been gone longer than PERSON_TIMEOUT_SECONDS.
        """
        events: list[Event] = []

        for person_id, movements in detections.items():
            person = self.people.get(person_id)
            if person is None:
                person = self.people[person_id] = PersonState(person_id, now)
            person.last_seen = now
            events += person.update(set(movements), now)

        for person_id, person in list(self.people.items()):
            if person_id in detections:
                continue
            # Their movements end the same way as if they'd stopped.
            events += person.update(set(), now)
            if now - person.last_seen > P.PERSON_TIMEOUT_SECONDS:
                del self.people[person_id]
                if self._on_person_removed:
                    self._on_person_removed(person_id)

        return events

    def active_movements(self) -> set[Movement]:
        """Every movement someone in the room is doing right now."""
        return set().union(*(p.active for p in self.people.values()))
