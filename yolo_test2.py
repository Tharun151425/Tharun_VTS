# 4.80 - 4.66 fps
from picamera2 import Picamera2
import cv2
import torch
import time

# Load YOLOv5 nano (lightweight and fast)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True)
model.to('cpu')

# Init camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Resize OpenCV window
cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 Detection", 1920, 1080)

# FPS tracking
prev_time = time.time()
frame_count = 0

while True:
    frame = picam2.capture_array()
    frame_count += 1

    # Inference
    with torch.no_grad():
        results = model(frame)
        result_frame = results.render()[0].copy()

    # Calculate FPS
    curr_time = time.time()
    fps = frame_count / (curr_time - prev_time)

    # Display FPS on frame
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv5 Detection", result_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
