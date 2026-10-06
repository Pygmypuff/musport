from ultralytics import YOLO


class YoloPersonDetector:
    """Wraps a YOLO pose model and tracks people."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self._last_results = None

    def detect(self, frame):
        """Track people and return persistent IDs and bounding boxes."""

        self._last_results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )

        boxes = self._last_results[0].boxes

        people = []

        for box in boxes:

            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            person_id = int(box.id[0])

            people.append({
                "id": person_id,
                "bbox": (x1, y1, x2, y2)
            })

        return people

    def annotate(self, frame):
        """Return the last frame's results drawn on the frame."""

        return self._last_results[0].plot()