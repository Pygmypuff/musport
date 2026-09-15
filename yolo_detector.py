from ultralytics import YOLO


class YoloPersonDetector:
    """Wraps a YOLO pose model and extracts per-person bounding boxes."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self._last_results = None

    def detect(self, frame):
        """Run YOLO on a frame and return a list of detected people.

        Each person is a dict: {"id": int, "bbox": (x1, y1, x2, y2)}
        """
        self._last_results = self.model(frame, verbose=False)
        boxes = self._last_results[0].boxes

        people = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people.append({"id": i, "bbox": (x1, y1, x2, y2)})

        return people

    def annotate(self, frame):
        """Return the last frame's results drawn on top of `frame`."""
        return self._last_results[0].plot()
