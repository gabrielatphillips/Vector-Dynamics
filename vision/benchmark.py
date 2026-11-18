
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from ultralytics import YOLO
import csv

BASE = Path(__file__).resolve().parent
IMAGE = BASE / "images" / "T1.jpeg"
OUTPUT = BASE / "results" / "benchmark.csv"

if not IMAGE.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE}")

model = YOLO("yolo11n.pt")

sizes = [640, 960, 1280]
repeats = 5

records = []

for size in sizes:
    print(f"\nTesting image size: {size}")

    # Warm up model at this image size
    model.predict(
        source=str(IMAGE),
        imgsz=size,
        conf=0.15,
        iou=0.40,
        agnostic_nms=True,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    times = []
    counts = []

    for run in range(repeats):
        start = perf_counter()

        results = model.predict(
            source=str(IMAGE),
            imgsz=size,
            conf=0.15,
            iou=0.40,
            agnostic_nms=True,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        elapsed_ms = (perf_counter() - start) * 1000
        vehicle_count = len(results[0].boxes)

        times.append(elapsed_ms)
        counts.append(vehicle_count)

        print(
            f"Run {run + 1}: "
            f"{elapsed_ms:.1f} ms, "
            f"{vehicle_count} detections"
        )

    average_ms = mean(times)
    median_ms = median(times)

    records.append({
        "image_size": size,
        "average_ms": round(average_ms, 2),
        "median_ms": round(median_ms, 2),
        "estimated_fps": round(1000 / average_ms, 2),
        "vehicle_detections": counts[-1]
    })

    print(f"Average: {average_ms:.1f} ms")
    print(f"Median: {median_ms:.1f} ms")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\nBenchmark saved to: {OUTPUT}")
