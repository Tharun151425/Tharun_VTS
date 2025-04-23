from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO
import csv

models = ["yolov8n.pt", "yolov8s.pt"]
resolutions = [(320, 240), (416, 312), (480, 360)]
imgszs = [128, 160, 192]

benchmark_duration = 10  # seconds per test
output_csv = "benchmark_results.csv"
output_txt = "benchmark_results.txt"

results = []

picam2 = Picamera2()
picam2.configure("preview")

cv2.namedWindow("YOLOv8 Benchmark", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Benchmark", 960, 720)

for model_path in models:
    model = YOLO(model_path)

    for res in resolutions:
        width, height = res
        picam2.preview_configuration.main.size = res
        picam2.preview_configuration.main.format = "RGB888"
        picam2.configure("preview")
        picam2.start()
        time.sleep(1)

        for imgsz in imgszs:
            print(f"\n?? Testing {model_path} @ {res} with imgsz={imgsz}")

            frame_times = []
            confs = []

            start_time = time.time()
            while True:
                loop_start = time.time()
                frame = picam2.capture_array()

                results_pred = model.predict(source=frame, imgsz=imgsz, conf=0.3, verbose=False)
                boxes = results_pred[0].boxes
                if boxes is not None and boxes.conf is not None:
                    confs.extend(boxes.conf.tolist())

                result_frame = results_pred[0].plot()
                cv2.imshow("YOLOv8 Benchmark", result_frame)

                loop_end = time.time()
                frame_times.append(loop_end - loop_start)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if loop_end - start_time >= benchmark_duration:
                    break

            if frame_times:
                fps_list = [1.0 / t for t in frame_times if t > 0]
                fps_min = min(fps_list)
                fps_max = max(fps_list)
                fps_avg = sum(fps_list) / len(fps_list)
            else:
                fps_min = fps_max = fps_avg = 0

            if confs:
                conf_min = min(confs)
                conf_max = max(confs)
                conf_avg = sum(confs) / len(confs)
            else:
                conf_min = conf_max = conf_avg = 0

            print(f"?? Result: FPS[min={fps_min:.2f}, max={fps_max:.2f}, avg={fps_avg:.2f}] | "
                  f"Conf[min={conf_min:.2f}, max={conf_max:.2f}, avg={conf_avg:.2f}]")

            results.append({
                "model": model_path,
                "resolution": f"{width}x{height}",
                "imgsz": imgsz,
                "fps_min": round(fps_min, 2),
                "fps_max": round(fps_max, 2),
                "fps_avg": round(fps_avg, 2),
                "conf_min": round(conf_min, 2),
                "conf_max": round(conf_max, 2),
                "conf_avg": round(conf_avg, 2),
            })

picam2.stop()
cv2.destroyAllWindows()

# Save to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Save to TXT
with open(output_txt, "w") as f:
    for r in results:
        f.write(f"{r['model']} | {r['resolution']} | imgsz={r['imgsz']} | "
                f"FPS[min={r['fps_min']}, max={r['fps_max']}, avg={r['fps_avg']}] | "
                f"Conf[min={r['conf_min']}, max={r['conf_max']}, avg={r['conf_avg']}]\n")

print(f"\n? Benchmark complete! Results saved to:\n- {output_csv}\n- {output_txt}")
