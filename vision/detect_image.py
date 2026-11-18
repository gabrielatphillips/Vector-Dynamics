
from pathlib import Path
from ultralytics import YOLO
import time
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "vision" / "images"
RESULTS_DIR = PROJECT_DIR / "vision" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    print("No images found.")
    raise SystemExit(1)

csv_file = RESULTS_DIR / "detections.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image",
        "vehicle_id",
        "vehicle_type",
        "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "width", "height",
        "processing_ms"
    ])

    for image_path in images:
        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            conf=0.15,
            imgsz=1280,
            iou=0.40,
            agnostic_nms=True,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        processing_ms = (time.perf_counter() - start) * 1000
        result = results[0]

        print(f"\nImage: {image_path.name}")
        print(f"Processing time: {processing_ms:.2f} ms")

        for vehicle_id, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls[0])
            vehicle_type = result.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            writer.writerow([
                image_path.name,
                vehicle_id,
                vehicle_type,
                round(confidence, 4),
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
                round(center_x, 2),
                round(center_y, 2),
                round(width, 2),
                round(height, 2),
                round(processing_ms, 2)
            ])

            print(
                f"Vehicle {vehicle_id}: {vehicle_type}, "
                f"confidence={confidence:.2%}, "
                f"center=({center_x:.1f}, {center_y:.1f})"
            )

        print(f"Total vehicles: {len(result.boxes)}")

        result.save(
            filename=str(RESULTS_DIR / image_path.name)
        )

print(f"\nCSV saved to: {csv_file}")
print("Detection complete!")
