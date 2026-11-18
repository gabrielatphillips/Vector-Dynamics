
from pathlib import Path
import csv
from collections import defaultdict

import numpy as np

# ----------------------------------
# File paths
# ----------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "vision" / "results" / "tracking_data.csv"
OUTPUT_CSV = ROOT / "tracking" / "kalman_results.csv"

# Kalman tuning parameters
PROCESS_ACCEL_STD = 1280.0  # pixels/s^2
MEASUREMENT_STD = 2.0     # pixels

# Reset filter if track disappears for too long
MAX_GAP_SECONDS = 0.5


class VehicleKalmanFilter:
    def __init__(self, x, y):

        # State: [x, y, vx, vy]
        self.state = np.array(
            [[x], [y], [0.0], [0.0]],
            dtype=float
        )

        # Initial uncertainty
        self.P = np.diag([
            100.0, 100.0,
            10000.0, 10000.0
        ])

        # Measurement matrix: only x,y observed
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.R = np.eye(2) * MEASUREMENT_STD**2

    def predict(self, dt):

        # Constant-velocity state transition
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # White-acceleration process noise
        G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])

        Q = (PROCESS_ACCEL_STD**2) * (G @ G.T)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_x, measured_y):

        z = np.array([
            [measured_x],
            [measured_y]
        ])

        # Innovation: measurement minus prediction
        innovation = z - self.H @ self.state

        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain (solve instead of inverse)
        K = np.linalg.solve(
            S, self.H @ self.P
        ).T

        self.state = self.state + K @ innovation

        # Joseph covariance update for stability
        I = np.eye(4)
        A = I - K @ self.H

        self.P = A @ self.P @ A.T + K @ self.R @ K.T

    def get_state(self):
        return self.state.flatten()


# ----------------------------------
# Load and organize tracking data
# ----------------------------------

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find tracking CSV: {INPUT_CSV}"
    )

observations = defaultdict(list)

with open(INPUT_CSV, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        track_id = int(row["track_id"])

        observations[track_id].append({
            "frame": int(row["frame"]),
            "timestamp": float(row["timestamp_s"]),
            "x": float(row["center_x"]),
            "y": float(row["center_y"])
        })


# ----------------------------------
# Process each vehicle independently
# ----------------------------------

output_rows = []
total_predictions = 0

for track_id, records in sorted(observations.items()):

    records.sort(key=lambda r: (r["frame"], r["timestamp"]))

    kf = None
    last_timestamp = None

    for record in records:

        t = record["timestamp"]
        measured_x = record["x"]
        measured_y = record["y"]

        # Initialize or reset a filter
        dt = None if last_timestamp is None else t - last_timestamp

        if (
            kf is None
            or dt is None
            or dt <= 0
            or dt > MAX_GAP_SECONDS
        ):
            kf = VehicleKalmanFilter(measured_x, measured_y)

            predicted_x = measured_x
            predicted_y = measured_y

            status = "initialized"
            dt_output = 0.0

        else:
            # Predict BEFORE seeing the new measurement
            kf.predict(dt)

            predicted_state = kf.get_state()
            predicted_x = predicted_state[0]
            predicted_y = predicted_state[1]

            # Correct using actual YOLO/ByteTrack position
            kf.update(measured_x, measured_y)

            status = "updated"
            dt_output = dt
            total_predictions += 1

        filtered = kf.get_state()

        prediction_error = np.hypot(
            measured_x - predicted_x,
            measured_y - predicted_y
        )

        output_rows.append({
            "frame": record["frame"],
            "timestamp_s": round(t, 4),
            "track_id": track_id,
            "status": status,
            "dt_s": round(dt_output, 4),
            "measured_x": round(measured_x, 3),
            "measured_y": round(measured_y, 3),
            "predicted_x": round(predicted_x, 3),
            "predicted_y": round(predicted_y, 3),
            "filtered_x": round(filtered[0], 3),
            "filtered_y": round(filtered[1], 3),
            "velocity_x_px_s": round(filtered[2], 3),
            "velocity_y_px_s": round(filtered[3], 3),
            "prediction_error_px": round(prediction_error, 3)
        })

        last_timestamp = t


# ----------------------------------
# Save results
# ----------------------------------

output_rows.sort(
    key=lambda r: (r["frame"], r["track_id"])
)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "frame",
    "timestamp_s",
    "track_id",
    "status",
    "dt_s",
    "measured_x",
    "measured_y",
    "predicted_x",
    "predicted_y",
    "filtered_x",
    "filtered_y",
    "velocity_x_px_s",
    "velocity_y_px_s",
    "prediction_error_px"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)


# ----------------------------------
# Performance summary
# ----------------------------------

errors = [
    row["prediction_error_px"]
    for row in output_rows
    if row["status"] == "updated"
]

print("\n========== KALMAN FILTER REPORT ==========")
print(f"Vehicle track IDs: {len(observations)}")
print(f"Total observations: {len(output_rows)}")
print(f"Predictions evaluated: {total_predictions}")

if errors:
    print(f"Mean prediction error: {np.mean(errors):.2f} px")
    print(f"Median prediction error: {np.median(errors):.2f} px")
    print(f"95th percentile error: {np.percentile(errors, 95):.2f} px")

print(f"\nSaved: {OUTPUT_CSV}")
print("==========================================")
