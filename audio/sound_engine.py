"""
Plays forest sounds.

Every sound has its own mixer channel, so the same sound never plays
twice at once.

- Held sounds loop while something holds them (set_held), fade in when
  they start and fade out and stop once nothing holds them.
- Triggered sounds (trigger) play once, and the trigger is ignored if
  that sound is already playing.
"""

from __future__ import annotations

import time
from random import choice

import pygame

from movements.enabled import BACKGROUND_SOUND_ENABLED
from movements.mapping import ALL_SOUNDS, BACKGROUND_SOUND, Sound, sound_path

# How long sounds take to fade in and out.
FADE_SECONDS = 0.5
CONTINUOUS_FADE_OUT_SECONDS = 1.5
# Channel volume of a held sound once fully faded in.
HELD_VOLUME = 0.8


class SoundChannel:
    """One sound on its own mixer channel."""

    def __init__(self, clips: list[pygame.mixer.Sound], channel_number: int):
        self.clips = clips
        self.channel = pygame.mixer.Channel(channel_number)
        self.held = False
        self.volume = 0.0
        self.volume_scale = 1.0
        self._looping = False
        self._oneshot = False
        self._oneshot_elapsed = 0.0
        self._oneshot_length = 0.0

    def trigger(self) -> bool:
        """Play once at full volume, unless this sound is already playing."""
        if self.channel.get_busy():
            return False
        clip = choice(self.clips)
        self._looping = False
        self._oneshot = True
        self._oneshot_elapsed = 0.0
        self._oneshot_length = clip.get_length()
        self.volume = 1.0
        self.channel.set_volume(self.volume)
        self.channel.play(clip)
        return True

    def update(self, dt: float):
        if self.held and not self.channel.get_busy():
            # Start the loop silent and fade it in.
            self._looping = True
            self._oneshot = False
            self.volume = 0.0
            self.channel.set_volume(0.0)
            self.channel.play(choice(self.clips), loops=-1)

        if self._oneshot:
            self._oneshot_elapsed += dt
            remaining = self._oneshot_length - self._oneshot_elapsed
            if remaining <= FADE_SECONDS:
                self.volume = max(0.0, remaining / FADE_SECONDS)
                self.channel.set_volume(self.volume)

            if remaining <= 0.0 or not self.channel.get_busy():
                self.channel.stop()
                self._oneshot = False
                self.volume = 0.0
            return

        if not self._looping:
            return

        target = HELD_VOLUME * self.volume_scale if self.held else 0.0
        fade_seconds = FADE_SECONDS if self.held else CONTINUOUS_FADE_OUT_SECONDS
        step = HELD_VOLUME * dt / fade_seconds
        if self.volume < target:
            self.volume = min(target, self.volume + step)
        else:
            self.volume = max(target, self.volume - step)
        self.channel.set_volume(self.volume)

        if not self.held and self.volume == 0.0:
            self.channel.stop()
            self._looping = False


class SoundEngine:
    """Owns the mixer: background bed plus one SoundChannel per sound."""

    def __init__(self, enable_background: bool = BACKGROUND_SOUND_ENABLED):
        """`enable_background` defaults to the switch in movements/enabled.py."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.set_num_channels(max(16, len(ALL_SOUNDS) + 1))

        # Channel 0 is the background bed; each sound gets its own after it.
        self._channels: dict[Sound, SoundChannel] = {
            sound: SoundChannel([self._load(f) for f in sound.files], number)
            for number, sound in enumerate(ALL_SOUNDS, start=1)
        }
        self._last_update = time.monotonic()

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

    def set_held(
        self, sounds: set[Sound], volume_scales: dict[Sound, float] | None = None
    ):
        """Set held sounds and optional per-sound volume multipliers."""
        volume_scales = volume_scales or {}
        for sound, channel in self._channels.items():
            channel.held = sound in sounds
            channel.volume_scale = max(0.0, min(1.0, volume_scales.get(sound, 1.0)))

    def trigger(self, sound: Sound) -> bool:
        """Play `sound` once. Returns False if it was already playing."""
        return self._channels[sound].trigger()

    def update(self):
        """Advance fades. Call once per frame."""
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        for channel in self._channels.values():
            channel.update(dt)

    def close(self):
        fade_ms = int(FADE_SECONDS * 1000)
        for channel in self._channels.values():
            if channel.channel.get_busy():
                channel.channel.fadeout(fade_ms)
        if self._background_channel.get_busy():
            self._background_channel.fadeout(fade_ms)

        # Let the mixer finish the fade instead of pygame.quit() cutting it off.
        time.sleep(FADE_SECONDS)
        pygame.mixer.stop()
