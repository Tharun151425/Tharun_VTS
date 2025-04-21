#
from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO

# Load YOLOv8n
model = YOLO("yolov8n.pt")  # You can try yolov8nano or yolov8s

# Init camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", 960, 720)

prev_time = time.time()
frame_count = 0

while True:
    frame = picam2.capture_array()
    frame_count += 1

    results = model.predict(source=frame, imgsz=32, conf=0.3, verbose=False)
    result_frame = results[0].plot()

    curr_time = time.time()
    fps = frame_count / (curr_time - prev_time)

    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
