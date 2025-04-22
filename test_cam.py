import cv2
from ultralytics import YOLO

# Load YOLOv8s-seg model
model = YOLO('yolov8n.pt')

# Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 segmentation
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("YOLOv8s-seg - Webcam", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
