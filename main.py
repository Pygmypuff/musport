"""
Entry point for the gesture-music prototype.

The display contains:
    - Full webcam feed on the left
    - A grid of cropped person views on the right
    - Each cropped view shows MediaPipe hand landmarks
      and the detected gesture

Run:
    python main.py

Press 'q' to quit.
"""

import cv2
import numpy as np

from gesture_pipeline import GesturePipeline
from yolo_detector import YoloPersonDetector


MODEL_PATH = "yolo26n-pose.pt"
MAIN_WINDOW = "Gesture Music Prototype"

NO_HANDS = "no hands"
UNKNOWN = "Unknown"

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

# Background color
BACKGROUND = (30, 30, 30)

# Space between person tiles
GRID_GAP = 10

# Border around person tiles
TILE_BORDER = (70, 70, 70)


def gesture_label(gestures):
    """Convert the GesturePipeline output into display text."""

    if not gestures:
        return NO_HANDS

    return ", ".join(gestures)


def resize_to_fit(image, width, height):
    """
    Resize an image so that it fits inside width x height
    while preserving its aspect ratio.
    """

    if image.size == 0:
        return None

    image_height, image_width = image.shape[:2]

    scale = min(
        width / image_width,
        height / image_height
    )

    new_width = max(1, int(image_width * scale))
    new_height = max(1, int(image_height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def create_person_tile(crop, person_id, gestures, tile_width, tile_height):
    """
    Create one tile for the person grid.

    The tile contains:
        - Gesture label at the top
        - Cropped person image
        - Person ID
    """

    # Create empty tile.
    tile = np.full(
        (tile_height, tile_width, 3),
        BACKGROUND,
        dtype=np.uint8
    )

    label = gesture_label(gestures)

    # ----------------------------------------------------------
    # Gesture information
    # ----------------------------------------------------------

    recognized = label not in (NO_HANDS, UNKNOWN)

    if recognized:
        color = (0, 255, 0)
    else:
        color = (0, 165, 255)

    # ----------------------------------------------------------
    # Resize crop to fit inside the tile.
    # ----------------------------------------------------------

    # Leave room at the top for the gesture label
    # and at the bottom for the person ID.
    top_height = 48
    bottom_height = 32

    image_area_height = (
        tile_height
        - top_height
        - bottom_height
    )

    resized = resize_to_fit(
        crop,
        tile_width - 12,
        image_area_height
    )

    if resized is not None:

        image_height, image_width = resized.shape[:2]

        # Center the crop inside the tile.
        x = (tile_width - image_width) // 2
        y = (
            top_height
            + (image_area_height - image_height) // 2
        )

        tile[
            y:y + image_height,
            x:x + image_width
        ] = resized

    # ----------------------------------------------------------
    # Gesture label
    # ----------------------------------------------------------

    # Dark bar behind the gesture text.
    cv2.rectangle(
        tile,
        (0, 0),
        (tile_width, top_height),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        tile,
        label,
        (12, 32),
        FONT,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

    # ----------------------------------------------------------
    # Person ID
    # ----------------------------------------------------------

    cv2.putText(
        tile,
        f"Person {person_id}",
        (10, tile_height - 10),
        FONT,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # ----------------------------------------------------------
    # Tile border
    # ----------------------------------------------------------

    cv2.rectangle(
        tile,
        (0, 0),
        (tile_width - 1, tile_height - 1),
        TILE_BORDER,
        1
    )

    return tile


def create_person_grid(person_tiles, grid_width, grid_height):
    """
    Arrange person tiles into a grid.

    Example with 4 people:

        ┌──────────┬──────────┐
        │ Person 1 │ Person 2 │
        ├──────────┼──────────┤
        │ Person 3 │ Person 4 │
        └──────────┴──────────┘
    """

    grid = np.full(
        (grid_height, grid_width, 3),
        BACKGROUND,
        dtype=np.uint8
    )

    if not person_tiles:

        text = "No people detected"

        text_size = cv2.getTextSize(
            text,
            FONT,
            0.7,
            2
        )[0]

        x = (grid_width - text_size[0]) // 2
        y = (grid_height + text_size[1]) // 2

        cv2.putText(
            grid,
            text,
            (x, y),
            FONT,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA
        )

        return grid

    # ----------------------------------------------------------
    # Calculate dynamic grid dimensions.
    # ----------------------------------------------------------

    rows = int(np.ceil(len(person_tiles) / GRID_COLUMNS))

    tile_width = (
        grid_width
        - GRID_GAP * (GRID_COLUMNS + 1)
    ) // GRID_COLUMNS

    tile_height = (
        grid_height
        - GRID_GAP * (rows + 1)
    ) // rows

    # ----------------------------------------------------------
    # Place each tile.
    # ----------------------------------------------------------

    for i, tile in enumerate(person_tiles):

        row = i // GRID_COLUMNS
        col = i % GRID_COLUMNS

        # Resize tile to exactly fit its cell.
        tile = cv2.resize(
            tile,
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA
        )

        x = (
            GRID_GAP
            + col * (tile_width + GRID_GAP)
        )

        y = (
            GRID_GAP
            + row * (tile_height + GRID_GAP)
        )

        grid[
            y:y + tile_height,
            x:x + tile_width
        ] = tile

    return grid


def main():

    detector = YoloPersonDetector(MODEL_PATH)
    pipeline = GesturePipeline()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # ------------------------------------------------------
            # YOLO detects people in the full webcam frame.
            # ------------------------------------------------------

            people = detector.detect(frame)

            person_tiles = []

            # ------------------------------------------------------
            # Process every detected person.
            # ------------------------------------------------------

            for person in people:

                x1, y1, x2, y2 = person["bbox"]

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Copy because GesturePipeline draws MediaPipe
                # landmarks directly onto the image it receives.
                display_crop = person_crop.copy()

                # --------------------------------------------------
                # MediaPipe hand detection + gesture classification
                # --------------------------------------------------

                gestures = pipeline.process(
                    person["id"],
                    display_crop
                )

                print(
                    f"Person {person['id']}: "
                    f"{gesture_label(gestures)}"
                )

                # --------------------------------------------------
                # Create a tile for this person.
                # --------------------------------------------------

                person_tiles.append(
                    create_person_tile(
                        display_crop,
                        person["id"],
                        gestures,
                        GRID_WIDTH // GRID_COLUMNS,
                        300
                    )
                )

            # ------------------------------------------------------
            # Annotate the full webcam frame with YOLO boxes.
            # ------------------------------------------------------

            annotated_frame = detector.annotate(frame)

            # ------------------------------------------------------
            # Create the right-side person grid.
            # ------------------------------------------------------

            grid_height = DISPLAY_HEIGHT

            person_grid = create_person_grid(
                person_tiles,
                GRID_WIDTH,
                grid_height
            )

            # ------------------------------------------------------
            # Resize full webcam image.
            # ------------------------------------------------------

            main_feed = resize_to_fit(
                annotated_frame,
                MAIN_WIDTH,
                DISPLAY_HEIGHT
            )

            # Create the left side.
            left_panel = np.full(
                (DISPLAY_HEIGHT, MAIN_WIDTH, 3),
                BACKGROUND,
                dtype=np.uint8
            )

            if main_feed is not None:

                feed_height, feed_width = main_feed.shape[:2]

                x = (MAIN_WIDTH - feed_width) // 2
                y = (DISPLAY_HEIGHT - feed_height) // 2

                left_panel[
                    y:y + feed_height,
                    x:x + feed_width
                ] = main_feed

            # ------------------------------------------------------
            # Combine left + right into ONE window.
            # ------------------------------------------------------

            combined = np.hstack(
                (left_panel, person_grid)
            )

            cv2.imshow(
                MAIN_WINDOW,
                combined
            )

            # ------------------------------------------------------
            # Quit with Q.
            # ------------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()