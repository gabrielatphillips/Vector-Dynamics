from pathlib import Path
from collections import defaultdict, deque
from time import perf_counter
import csv

import cv2
from ultralytics import YOLO

# -----------------------------------
# Project configuration
# -----------------------------------

BASE = Path(__file__).resolve().parent

VIDEO = BASE / "videos" / "road_test.mp4"
OUTPUT = BASE / "results" / "road_tracked.mp4"
CSV_OUTPUT = BASE / "results" / "tracking_data.csv"

MODEL = "yolo11n.pt"

IMAGE_SIZE = 960
CONFIDENCE = 0.15
IOU = 0.40

VEHICLE_CLASSES = [2, 3, 5, 7]

# Number of previous positions displayed
TRAIL_LENGTH = 30

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Initialize YOLO and video
# -----------------------------------

model = YOLO(MODEL)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video")

# Store previous positions for each vehicle
track_history = defaultdict(
    lambda: deque(maxlen=TRAIL_LENGTH)
)

frame_number = 0
total_processing_time = 0.0
unique_track_ids = set()

# -----------------------------------
# Process video and save CSV
# -----------------------------------

try:
    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "frame",
            "timestamp_s",
            "track_id",
            "vehicle_type",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "width",
            "height",
            "processing_ms"
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            timestamp = (frame_number - 1) / fps

            # -----------------------------------
            # Run YOLO + ByteTrack
            # -----------------------------------

            start_time = perf_counter()

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=IOU,
                agnostic_nms=True,
                classes=VEHICLE_CLASSES,
                verbose=False
            )

            processing_ms = (
                perf_counter() - start_time
            ) * 1000

            total_processing_time += processing_ms

            result = results[0]

            # Draw YOLO bounding boxes and IDs
            annotated = result.plot()

            boxes = result.boxes

            # -----------------------------------
            # Record tracked vehicles
            # -----------------------------------

            if boxes is not None and boxes.id is not None:

                for box in boxes:

                    track_id = int(box.id.item())

                    class_id = int(box.cls.item())
                    vehicle_type = result.names[class_id]

                    confidence = float(box.conf.item())

                    x1, y1, x2, y2 = map(
                        float,
                        box.xyxy[0].tolist()
                    )

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    box_width = x2 - x1
                    box_height = y2 - y1

                    unique_track_ids.add(track_id)

                    # Save position history
                    history = track_history[track_id]

                    # Reset trail after a long gap
                    if history:
                        last_frame = history[-1][0]
                        if frame_number - last_frame > 5:
                            history.clear()

                    history.append((
                        frame_number,
                        int(center_x),
                        int(center_y)
                    ))

                    # Save detection to CSV
                    csv_writer.writerow([
                        frame_number,
                        round(timestamp, 4),
                        track_id,
                        vehicle_type,
                        round(confidence, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        round(center_x, 2),
                        round(center_y, 2),
                        round(box_width, 2),
                        round(box_height, 2),
                        round(processing_ms, 2)
                    ])

                    # -----------------------------------
                    # Draw movement trail
                    # -----------------------------------

                    points = [
                        (x, y)
                        for _, x, y in history
                    ]

                    for i in range(1, len(points)):

                        cv2.line(
                            annotated,
                            points[i - 1],
                            points[i],
                            (0, 255, 255),
                            2
                        )

                    # Draw vehicle center
                    cv2.circle(
                        annotated,
                        (int(center_x), int(center_y)),
                        4,
                        (0, 0, 255),
                        -1
                    )

            # -----------------------------------
            # Display system information
            # -----------------------------------

            cv2.putText(
                annotated,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Processing: {processing_ms:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            writer.write(annotated)

            if frame_number % 100 == 0:
                print(
                    f"Frame {frame_number} | "
                    f"Tracks created: {len(unique_track_ids)} | "
                    f"Processing: {processing_ms:.1f} ms"
                )

finally:
    cap.release()
    writer.release()

# -----------------------------------
# Final performance report
# -----------------------------------

print("\n========== TRACKING REPORT ==========")

print(f"Frames processed: {frame_number}")

print(f"Unique track IDs created: {len(unique_track_ids)}")

if frame_number > 0:

    average_ms = (
        total_processing_time / frame_number
    )

    print(f"Average processing: {average_ms:.2f} ms")

    print(
        f"Estimated processing FPS: "
        f"{1000 / average_ms:.2f}"
    )

print(f"\nVideo saved: {OUTPUT}")
print(f"CSV saved: {CSV_OUTPUT}")

print("=====================================")
