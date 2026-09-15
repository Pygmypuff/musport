from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detects people in the frame
    results = model(frame, verbose=False)

    # List of people detected in this frame
    people = []

    # Extract each detected person's bounding box
    boxes = results[0].boxes

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        person = {
            "id": i,
            "bbox": (x1, y1, x2, y2) # Get the box coordinates to be passed to mediapipe
        }

        people.append(person)

        # -------------------------------------------------
        # MEDIAPIPE PIPELINE STARTS HERE
        #
        # Crop this person's image using their bounding box:
        #
        # person_crop = frame[y1:y2, x1:x2]
        #
        # Then feed person_crop to MediaPipe:
        #
        # gesture = mediapipe.detect(person_crop)
        #
        # Then send the gesture to the music engine.
        # -------------------------------------------------

    # Show YOLO's detections for now
    annotated_frame = results[0].plot()

    cv2.imshow("Gesture Music Prototype", annotated_frame) # show the camera when the app is running

    # Quit the program by pressing 'Q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()