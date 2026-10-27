"""
Keyboard demo for the forest sound bed (no webcam).

Use this to audition movement → sound pairings before detectors are ready.

Run:
    python demo_sounds.py

Keys match movements/mapping.py (B, 1–5, C). Press Q to quit.
"""

import pygame

from audio.sound_engine import SoundEngine
from movements.mapping import MOVEMENT_SOUNDS, Movement


def key_map() -> dict[int, Movement]:
    """Pygame key code → movement, from the keys in mapping.py.

    Call after pygame.init(); key_code() needs it.
    """
    return {
        pygame.key.key_code(spec.keyboard): movement
        for movement, spec in MOVEMENT_SOUNDS.items()
    }


def main():
    pygame.init()
    screen = pygame.display.set_mode((520, 320))
    pygame.display.set_caption("Musport — sound demo")
    clock = pygame.time.Clock()
    engine = SoundEngine(enable_background=True)
    key_to_movement = key_map()

    print("Musport sound demo")
    print("Forest background is playing.")
    for movement, spec in MOVEMENT_SOUNDS.items():
        print(f"  [{spec.keyboard}]  {spec.sound.name:8} ← {spec.label}")
    print("  [q]  quit")

    held_continuous: set[Movement] = set()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        continue

                    movement = key_to_movement.get(event.key)
                    if movement is None:
                        continue

                    spec = MOVEMENT_SOUNDS[movement]
                    print(f"→ {spec.sound.name} ({spec.label})")
                    if spec.playback == "continuous":
                        held_continuous.add(movement)
                    else:
                        engine.trigger(spec.sound)

                elif event.type == pygame.KEYUP:
                    movement = key_to_movement.get(event.key)
                    held_continuous.discard(movement)

            engine.set_held(
                {MOVEMENT_SOUNDS[m].sound for m in held_continuous}
            )
            engine.update()
            screen.fill((24, 36, 28))
            pygame.display.flip()
            clock.tick(60)
    finally:
        engine.close()
        pygame.quit()


if __name__ == "__main__":
    main()
