from picamera2 import Picamera2
import cv2
import time
import numpy as np
from ultralytics import YOLO

import pandas as pd
import os

from datetime import datetime

# === Append to res_yolo.txt ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_filename = "res_yolo.txt"

# Define CSV file path
csv_file_path = "./res_yolo.csv"

# === CONFIGURATION ===
MODEL_NAME = "yolov8m.pt"  # Change to yolov8s.pt or yolov8n-seg.pt
IMG_SIZE = 160
CONF_THRESH = 0.3
RESOLUTION = (480, 360)

# === INIT ===
model = YOLO(MODEL_NAME)

picam2 = Picamera2()
picam2.preview_configuration.main.size = RESOLUTION
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", 960, 720)

# === METRICS STORAGE ===
fps_list = []
conf_list = []
frame_count = 0
start_time = time.time()

# Simulated Confusion Matrix Components
total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0  # TN is hard to evaluate live without labels

while True:
    frame = picam2.capture_array()
    frame_count += 1

    results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
    result_frame = results[0].plot()
    detections = results[0].boxes

    # Track confidence scores
    if detections:
        for box in detections:
            conf_list.append(float(box.conf[0]))
    if frame_count == 200:
        break

    # Simulated Confusion Matrix (assumes one real object per frame)
    detected = len(detections)
    if detected > 0:
        total_tp += 1
        total_fp += max(0, detected - 1)  # Assume 1 true object, rest are FP
    else:
        total_fn += 1

    # FPS tracking
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    fps_list.append(fps)

    # Show FPS on frame
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === POST INFERENCE STATS ===
cv2.destroyAllWindows()
end_time = time.time()
total_time = end_time - start_time

# === CONFIDENCE STATS ===
avg_conf = np.mean(conf_list) if conf_list else 0
min_conf = np.min(conf_list) if conf_list else 0
max_conf = np.max(conf_list) if conf_list else 0

# === FPS STATS ===
avg_fps = np.mean(fps_list)
min_fps = np.min(fps_list)
max_fps = np.max(fps_list)

# === METRIC CALCULATIONS ===
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) else 0
npv = total_tn / (total_tn + total_fn) if (total_tn + total_fn) else 0
accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

remark = input("How is it :")
summary_text = f"""
====================== YOLOv8 TEST SUMMARY ======================
Timestamp          : {timestamp}
Model Used         : {MODEL_NAME}
Image Size         : {IMG_SIZE}
Camera Resolution  : {RESOLUTION}
Confidence Thresh  : {CONF_THRESH}
Total Time (s)     : {total_time:.2f}
Total Frames       : {frame_count}

--- FPS Stats ---
Avg FPS            : {avg_fps:.2f}
Max FPS            : {max_fps:.2f}
Min FPS            : {min_fps:.2f}

--- Detection Confidence ---
Avg Confidence     : {avg_conf:.2f}
Max Confidence     : {max_conf:.2f}
Min Confidence     : {min_conf:.2f}

--- ML Performance Metrics (Simulated) ---
Precision          : {precision:.2f}
Recall (Sensitivity): {recall:.2f}
Specificity        : {specificity:.2f}
NPV                : {npv:.2f}
Accuracy           : {accuracy:.2f}
F1 Score           : {f1:.2f}
Confusion Matrix   : TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}
Remark : {remark}
===============================================================\n\n"""

# Append to file
with open(log_filename, "a") as f:
    f.write(summary_text)

# === PRINT SUMMARY === print("\n====================== YOLOv8 TEST SUMMARY ======================")
print(f"""
====================== YOLOv8 TEST SUMMARY ======================
Timestamp          : {timestamp}

Avg FPS            : {avg_fps:.2f}
Max Confidence     : {max_conf:.2f}
Min Confidence     : {min_conf:.2f}
Precision          : {precision:.2f}
Recall (Sensitivity): {recall:.2f}
{MODEL_NAME} {IMG_SIZE} {RESOLUTION} {CONF_THRESH}
===============================================================\n\n""")

print(f"\n📝 Summary appended to: {log_filename}")
# Create result row dictionary
result_row = {
    "Model": MODEL_NAME,
    "Resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}",
    "Image Size": IMG_SIZE,
    "Conf Thresh": CONF_THRESH,
    "Min FPS": round(min_fps, 2),
    "Max FPS": round(max_fps, 2),
    "Avg FPS": round(avg_fps, 2),
    "Min Conf": round(min_conf, 2),
    "Max Conf": round(max_conf, 2),
    "Avg Conf": round(avg_conf, 2),
    "Precision": round(precision, 2),
    "Recall": round(recall, 2),
    "Accuracy": round(accuracy, 2),
    "F1 Score": round(f1, 2),
    "Remark": remark
}

# Convert to DataFrame
df = pd.DataFrame([result_row])

# Append to CSV
if os.path.exists(csv_file_path):
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file_path, index=False)

print(f"\n📁 Results saved to: {csv_file_path}")
