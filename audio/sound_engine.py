"""
Plays forest sounds for active yoga movements.

Continuous sounds (breeze, rain, wind, water) fade volume toward a
target while the movement is held. One-shot sounds (birds, leaves,
cricket) fire once per fresh detection.
"""

from random import choice

import pygame

from movements.mapping import (
    BACKGROUND_SOUND,
    MOVEMENT_SOUNDS,
    Movement,
    sound_path,
)


class ContinuousSound:
    def __init__(self, sound: pygame.mixer.Sound, channel_number: int):
        self.sound = sound
        self.channel = pygame.mixer.Channel(channel_number)
        self.current_volume = 0.0
        self.target_volume = 0.0
        self._started = False

    def ensure_playing(self):
        if not self._started or not self.channel.get_busy():
            self.channel.set_volume(self.current_volume)
            self.channel.play(self.sound, loops=-1)
            self._started = True

    def set_intensity(self, intensity: float):
        self.target_volume = max(0.0, min(1.0, intensity))
        if intensity > 0:
            self.ensure_playing()

    def update(self):
        self.current_volume += (self.target_volume - self.current_volume) * 0.03
        if self._started:
            self.channel.set_volume(self.current_volume)


class OneShotSound:
    def __init__(self, sounds: list[pygame.mixer.Sound], channel_number: int):
        self.sounds = sounds
        self.channel = pygame.mixer.Channel(channel_number)

    def play(self):
        """Play one of the variations, picked at random."""
        self.channel.play(choice(self.sounds))


class SoundEngine:
    """Map active Movement sets → mixer channels."""

    def __init__(self, enable_background: bool = True):
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.set_num_channels(16)

        self._continuous: dict[Movement, ContinuousSound] = {}
        self._oneshots: dict[Movement, OneShotSound] = {}
        self._prev_active: set[Movement] = set()

        # Channel 0 is the background bed; each movement gets its own after it.
        for channel_num, (movement, spec) in enumerate(
            MOVEMENT_SOUNDS.items(), start=1
        ):
            sounds = [self._load(name) for name in spec.files]
            if spec.playback == "continuous":
                self._continuous[movement] = ContinuousSound(
                    sounds[0], channel_num
                )
            else:
                self._oneshots[movement] = OneShotSound(sounds, channel_num)

        self._background_channel = pygame.mixer.Channel(0)
        if enable_background:
            bg = self._load(BACKGROUND_SOUND)
            bg.set_volume(0.2)
            self._background_channel.play(bg, loops=-1)

    def _load(self, filename: str) -> pygame.mixer.Sound:
        path = sound_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing sound file: {path}")
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(0.8)
        return sound

    def _play_oneshot(self, movement: Movement):
        oneshot = self._oneshots.get(movement)
        if oneshot:
            oneshot.play()

    def set_active(self, active: set[Movement], intensity: float = 0.8):
        """Update playback from the set of movements currently detected."""
        for movement, continuous in self._continuous.items():
            continuous.set_intensity(intensity if movement in active else 0.0)

        for movement in active - self._prev_active:
            self._play_oneshot(movement)

        self._prev_active = set(active)

    def trigger(self, movement: Movement, intensity: float = 0.8):
        """Manual / keyboard trigger for one movement."""
        if movement in self._continuous:
            self._continuous[movement].set_intensity(intensity)
        else:
            self._play_oneshot(movement)

    def update(self):
        for continuous in self._continuous.values():
            continuous.update()

    def close(self):
        pygame.mixer.stop()
