
from pathlib import Path
import cv2
import time
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
VIDEO = BASE / "videos" / "road_test.mp4"
OUTPUT = BASE / "results" / "road_detected.mp4"

if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO}")

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

frame_number = 0
total_time = 0.0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=960,
            conf=0.15,
            iou=0.40,
            agnostic_nms=True,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        elapsed = time.perf_counter() - start
        total_time += elapsed
        frame_number += 1

        result = results[0]
        annotated = result.plot()

        writer.write(annotated)

        if frame_number % 30 == 0:
            print(
                f"Frame {frame_number} | "
                f"Vehicles: {len(result.boxes)} | "
                f"Inference: {elapsed * 1000:.1f} ms"
            )

finally:
    cap.release()
    writer.release()

if frame_number:
    average_ms = total_time / frame_number * 1000
    print(f"\nProcessed {frame_number} frames")
    print(f"Average prediction time: {average_ms:.1f} ms")
    print(f"Saved video: {OUTPUT}")
else:
    print("No frames processed.")
