from ultralytics import YOLO
import cv2
import numpy as np

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

    crops = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        person = {
            "id": i,
            "bbox": (x1, y1, x2, y2)
        }

        people.append(person)

        # Crop the person from the original frame
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size > 0:
            # Resize the crop so all person previews are the same size
            person_crop = cv2.resize(person_crop, (200, 250))

            # Show the person's ID
            cv2.putText(
                person_crop,
                f"Person {i}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            crops.append(person_crop)

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
    annotated_frame = results[0].plot() # show the camera when the app is running
    annotated_frame = cv2.resize(annotated_frame, (640, 480))

    # Create an area for the individual person crops
    side_panel = np.zeros((480, 400, 3), dtype=np.uint8)

    # Put each person crop into the side panel
    for i, crop in enumerate(crops):
        x = (i % 2) * 200
        y = (i // 2) * 250

        if y + 250 <= 480:
            side_panel[y:y + 250, x:x + 200] = crop

    # Combine the camera view and person crops into one window
    display = np.hstack((annotated_frame, side_panel))

    cv2.imshow("Gesture Music Prototype", display)

    # Quit the program by pressing 'Q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()