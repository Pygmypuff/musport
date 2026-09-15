from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Gesture Music Prototype", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()