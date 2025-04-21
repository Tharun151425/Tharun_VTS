from picamera2 import Picamera2
import cv2
import torch

# Load YOLOv5 nano (fastest)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True)
model.to('cpu')

# Init camera
picam2 = Picamera2()
#picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()

    with torch.no_grad():
        results = model(frame)

    cv2.imshow("YOLOv5 Detection", results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
