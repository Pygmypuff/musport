"""
Canonical map: senior-living yoga movement → forest sound.

This is the single source of truth for what the class should hear
when a movement is recognized. Detectors return Movement IDs from
here; the sound engine looks up files and playback style from here.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


SOUNDS_DIR = Path(__file__).resolve().parent.parent / "assets" / "sounds"


class Movement(str, Enum):
    SIDE_BEND = "side_bend"  # one hand up, torso leaning
    NECK_TURN = "neck_turn"  # head / neck rotation
    BEND_DOWN = "bend_down"  # forward fold
    LEG_EXTENSION = "leg_extension"
    TORSO_TURN = "torso_turn"
    HEEL_RAISE = "heel_raise"  # seated heel raise
    FINGER_EXTENSION = "finger_extension"


Playback = Literal["oneshot", "continuous"]


@dataclass(frozen=True)
class SoundSpec:
    """How a movement's sound should play."""

    label: str  # human-readable movement name
    sound: str  # nature sound name shown in the UI
    files: tuple[str, ...]  # wav filenames under assets/sounds/
    playback: Playback
    keyboard: str  # key used in demo_sounds.py


MOVEMENT_SOUNDS: dict[Movement, SoundSpec] = {
    Movement.SIDE_BEND: SoundSpec(
        label="Side bend (one hand up)",
        sound="Breeze",
        files=("breeze.wav",),
        playback="continuous",
        keyboard="b",
    ),
    Movement.NECK_TURN: SoundSpec(
        label="Neck / head turn",
        sound="Birds",
        files=("birds1.wav", "birds2.wav", "birds3.wav"),
        playback="oneshot",
        keyboard="1",
    ),
    Movement.BEND_DOWN: SoundSpec(
        label="Bend down",
        sound="Rain",
        files=("long_rain.wav",),
        playback="continuous",
        keyboard="2",
    ),
    Movement.LEG_EXTENSION: SoundSpec(
        label="Leg extension",
        sound="Wind",
        files=("wind_blowing.wav",),
        playback="continuous",
        keyboard="3",
    ),
    Movement.TORSO_TURN: SoundSpec(
        label="Torso turn",
        sound="Water",
        files=("water_flowing.wav",),
        playback="continuous",
        keyboard="4",
    ),
    Movement.HEEL_RAISE: SoundSpec(
        label="Seated heel raise",
        sound="Leaves",
        files=("stepping_on_leaves.wav",),
        playback="oneshot",
        keyboard="5",
    ),
    Movement.FINGER_EXTENSION: SoundSpec(
        label="Finger extension",
        sound="Cricket",
        files=("cricket.wav",),
        playback="oneshot",
        keyboard="c",
    ),
}

BACKGROUND_SOUND = "forest_background.wav"


def sound_path(filename: str) -> Path:
    return SOUNDS_DIR / filename


def display_name(movement: Movement) -> str:
    """Short UI label: 'Breeze — Side bend (one hand up)'."""
    spec = MOVEMENT_SOUNDS[movement]
    return f"{spec.sound} — {spec.label}"
