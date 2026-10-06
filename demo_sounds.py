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


KEY_TO_MOVEMENT = {
    pygame.key.key_code(spec.keyboard): movement
    for movement, spec in MOVEMENT_SOUNDS.items()
}


def main():
    pygame.init()
    screen = pygame.display.set_mode((520, 320))
    pygame.display.set_caption("Musport — sound demo")
    clock = pygame.time.Clock()
    engine = SoundEngine(enable_background=True)

    print("Musport sound demo")
    print("Forest background is playing.")
    for movement, spec in MOVEMENT_SOUNDS.items():
        print(f"  [{spec.keyboard}]  {spec.sound:8} ← {spec.label}")
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

                    movement = KEY_TO_MOVEMENT.get(event.key)
                    if movement is None:
                        continue

                    spec = MOVEMENT_SOUNDS[movement]
                    print(f"→ {spec.sound} ({spec.label})")
                    if spec.playback == "continuous":
                        held_continuous.add(movement)
                        engine.set_active(held_continuous)
                    else:
                        engine.trigger(movement)

                elif event.type == pygame.KEYUP:
                    movement = KEY_TO_MOVEMENT.get(event.key)
                    if movement and movement in held_continuous:
                        held_continuous.discard(movement)
                        engine.set_active(held_continuous)

            engine.update()
            screen.fill((24, 36, 28))
            pygame.display.flip()
            clock.tick(60)
    finally:
        engine.close()
        pygame.quit()


if __name__ == "__main__":
    main()
