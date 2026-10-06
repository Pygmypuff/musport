"""
Window layout and drawing for the Musport display.

The display contains:
    - Full webcam feed on the left
    - A grid of cropped person views on the right
    - Each cropped view shows MediaPipe hand landmarks
      and the detected yoga movement / sound

main.py only calls render(); everything else here is a drawing helper.
"""

import cv2
import numpy as np

from movements.mapping import Movement, display_name


WINDOW_NAME = "Musport Yoga"

NOTHING_DETECTED = "nothing detected"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Size of the final combined window
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Width allocated to the full webcam feed
MAIN_WIDTH = 800

# Width allocated to the person grid
GRID_WIDTH = DISPLAY_WIDTH - MAIN_WIDTH

# Number of columns in the person grid
GRID_COLUMNS = 2

# Space between person tiles
GRID_GAP = 10

# Colors (BGR)
BACKGROUND = (30, 30, 30)
TILE_BORDER = (70, 70, 70)
LABEL_BAR = (20, 20, 20)
MUTED_TEXT = (180, 180, 180)
RECOGNIZED = (0, 255, 0)
UNRECOGNIZED = (0, 165, 255)


def gesture_label(movements: list[Movement]) -> str:
    """Convert detected Movement IDs into display text."""
    if not movements:
        return NOTHING_DETECTED

    return ", ".join(display_name(m) for m in movements)


def blank(width, height):
    """Empty canvas in the background color."""
    return np.full((height, width, 3), BACKGROUND, dtype=np.uint8)


def resize_to_fit(image, width, height):
    """
    Resize an image so that it fits inside width x height
    while preserving its aspect ratio.
    """
    if image.size == 0:
        return None

    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)

    new_width = max(1, int(image_width * scale))
    new_height = max(1, int(image_height * scale))

    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )


def paste_centered(canvas, image, left, top, width, height):
    """
    Scale `image` to fit the width x height area of `canvas` whose
    top-left corner is (left, top), and draw it centered in that area.
    """
    resized = resize_to_fit(image, width, height)
    if resized is None:
        return

    image_height, image_width = resized.shape[:2]
    x = left + (width - image_width) // 2
    y = top + (height - image_height) // 2
    canvas[y:y + image_height, x:x + image_width] = resized


def grid_layout(person_count):
    """
    Rows in the grid and the size of one cell for the given number of
    people.

    Tiles are built at this size rather than a fixed one, so they are
    never stretched or squashed when placed into the grid.
    """
    rows = max(1, int(np.ceil(person_count / GRID_COLUMNS)))

    tile_width = (GRID_WIDTH - GRID_GAP * (GRID_COLUMNS + 1)) // GRID_COLUMNS
    tile_height = (DISPLAY_HEIGHT - GRID_GAP * (rows + 1)) // rows

    return rows, tile_width, tile_height


def create_person_tile(crop, person_id, movements, tile_width, tile_height):
    """
    Create one tile for the person grid.

    The tile contains:
        - Pose / gesture label at the top
        - Cropped person image
        - Person ID
    """
    tile = blank(tile_width, tile_height)

    label = gesture_label(movements)
    color = RECOGNIZED if movements else UNRECOGNIZED

    # Leave room at the top for the gesture label and at the bottom for
    # the person ID. Both shrink with the tile so that a crowd, which
    # makes every tile short, does not leave the crop with no room.
    top_height = min(48, max(18, tile_height // 4))
    bottom_height = min(32, max(12, tile_height // 6))
    image_area_height = tile_height - top_height - bottom_height

    paste_centered(
        tile, crop, 6, top_height, tile_width - 12, image_area_height
    )

    # Dark bar behind the gesture text.
    cv2.rectangle(tile, (0, 0), (tile_width, top_height), LABEL_BAR, -1)

    # Size the text to the tile: labels combining a pose with two hand
    # gestures are far wider than a tile at a fixed scale, and would clip.
    (unit_width, _), _ = cv2.getTextSize(label, FONT, 1.0, 1)
    label_scale = min(
        0.65,
        (tile_width - 24) / unit_width,
        0.65 * top_height / 48,
    )
    label_thickness = 2 if label_scale >= 0.5 else 1

    cv2.putText(
        tile,
        label,
        (12, int(top_height * 0.68)),
        FONT,
        label_scale,
        color,
        label_thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        f"Person {person_id}",
        (10, tile_height - 10),
        FONT,
        0.45,
        MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        tile, (0, 0), (tile_width - 1, tile_height - 1), TILE_BORDER, 1
    )

    return tile


def create_person_grid(person_views):
    """
    Build one tile per (person_id, crop, movements) and arrange them
    into a grid.

    Example with 4 people:

        ┌──────────┬──────────┐
        │ Person 1 │ Person 2 │
        ├──────────┼──────────┤
        │ Person 3 │ Person 4 │
        └──────────┴──────────┘
    """
    grid = blank(GRID_WIDTH, DISPLAY_HEIGHT)

    if not person_views:
        text = "No people detected"
        text_width, text_height = cv2.getTextSize(text, FONT, 0.7, 2)[0]
        x = (GRID_WIDTH - text_width) // 2
        y = (DISPLAY_HEIGHT + text_height) // 2
        cv2.putText(grid, text, (x, y), FONT, 0.7, MUTED_TEXT, 2, cv2.LINE_AA)
        return grid

    _, tile_width, tile_height = grid_layout(len(person_views))

    for i, (person_id, crop, movements) in enumerate(person_views):
        row = i // GRID_COLUMNS
        col = i % GRID_COLUMNS

        x = GRID_GAP + col * (tile_width + GRID_GAP)
        y = GRID_GAP + row * (tile_height + GRID_GAP)

        grid[y:y + tile_height, x:x + tile_width] = create_person_tile(
            crop, person_id, movements, tile_width, tile_height
        )

    return grid


def render(annotated_frame, person_views):
    """
    Build the full window image: webcam feed on the left, person grid
    on the right. `person_views` is a list of (person_id, crop, movements).
    """
    left_panel = blank(MAIN_WIDTH, DISPLAY_HEIGHT)
    paste_centered(
        left_panel, annotated_frame, 0, 0, MAIN_WIDTH, DISPLAY_HEIGHT
    )

    return np.hstack((left_panel, create_person_grid(person_views)))
