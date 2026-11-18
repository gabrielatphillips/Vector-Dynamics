
from pathlib import Path
from collections import defaultdict
import csv
import itertools
import numpy as np

# ==========================================
# VECTOR DYNAMICS - KALMAN AUTO TUNER
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "vision" / "results" / "tracking_data.csv"
OUTPUT_DIR = ROOT / "tracking" / "tuning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test different acceleration and measurement noise
PROCESS_NOISE_VALUES = [20, 40, 80, 160, 320, 640, 1280]
MEASUREMENT_NOISE_VALUES = [2, 4, 8, 16, 32, 64]

MAX_GAP_SECONDS = 0.5
TRAIN_FRACTION = 0.70

# Fixed initial uncertainty from original filter
INITIAL_POSITION_VARIANCE = 100.0
INITIAL_VELOCITY_VARIANCE = 10000.0


# ==========================================
# KALMAN FILTER
# ==========================================

class KalmanFilter:
    def __init__(self, x, y, process_std, measurement_std):

        self.state = np.array(
            [x, y, 0.0, 0.0],
            dtype=float
        )

        self.P = np.diag([
            INITIAL_POSITION_VARIANCE,
            INITIAL_POSITION_VARIANCE,
            INITIAL_VELOCITY_VARIANCE,
            INITIAL_VELOCITY_VARIANCE
        ])

        self.process_std = process_std

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.R = np.eye(2) * measurement_std**2

    def predict(self, dt):

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ], dtype=float)

        Q = self.process_std**2 * (G @ G.T)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, x, y):

        z = np.array([x, y], dtype=float)

        innovation = z - self.H @ self.state

        S = self.H @ self.P @ self.H.T + self.R

        K = np.linalg.solve(
            S, self.H @ self.P
        ).T

        self.state = self.state + K @ innovation

        I = np.eye(4)
        A = I - K @ self.H

        self.P = A @ self.P @ A.T + K @ self.R @ K.T


# ==========================================
# LOAD TRACKING DATA
# ==========================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Tracking data not found: {INPUT_CSV}"
    )

tracks = defaultdict(list)

with open(INPUT_CSV, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        track_id = int(row["track_id"])

        tracks[track_id].append({
            "frame": int(row["frame"]),
            "time": float(row["timestamp_s"]),
            "x": float(row["center_x"]),
            "y": float(row["center_y"])
        })

for records in tracks.values():
    records.sort(key=lambda r: (r["frame"], r["time"]))

# Deterministic split by complete track IDs
track_ids = sorted(tracks.keys())

rng = np.random.default_rng(42)
rng.shuffle(track_ids)

split_index = int(len(track_ids) * TRAIN_FRACTION)

train_ids = track_ids[:split_index]
test_ids = track_ids[split_index:]

print(f"Training tracks: {len(train_ids)}")
print(f"Held-out test tracks: {len(test_ids)}")


# ==========================================
# EVALUATE ONE PARAMETER COMBINATION
# ==========================================

def evaluate(track_subset, process_std, measurement_std):

    kalman_errors = []
    baseline_errors = []

    for track_id in track_subset:

        records = tracks[track_id]

        kf = None
        previous = None

        for record in records:

            t = record["time"]
            x = record["x"]
            y = record["y"]

            if previous is None:
                dt = None
            else:
                dt = t - previous["time"]

            if (
                kf is None
                or dt is None
                or dt <= 0
                or dt > MAX_GAP_SECONDS
            ):
                kf = KalmanFilter(
                    x, y,
                    process_std,
                    measurement_std
                )

            else:
                # Predict without using current measurement
                kf.predict(dt)

                prediction = kf.state[:2].copy()

                kalman_error = np.hypot(
                    x - prediction[0],
                    y - prediction[1]
                )

                baseline_error = np.hypot(
                    x - previous["x"],
                    y - previous["y"]
                )

                kalman_errors.append(kalman_error)
                baseline_errors.append(baseline_error)

                # Update after evaluating prediction
                kf.update(x, y)

            previous = record

    if not kalman_errors:
        raise RuntimeError("No valid predictions to evaluate.")

    return {
        "kalman_mean": float(np.mean(kalman_errors)),
        "kalman_median": float(np.median(kalman_errors)),
        "kalman_p95": float(np.percentile(kalman_errors, 95)),
        "baseline_mean": float(np.mean(baseline_errors)),
        "baseline_median": float(np.median(baseline_errors)),
        "count": len(kalman_errors)
    }


# ==========================================
# GRID SEARCH ON TRAINING TRACKS
# ==========================================

results = []

combinations = list(itertools.product(
    PROCESS_NOISE_VALUES,
    MEASUREMENT_NOISE_VALUES
))

print(f"\nTesting {len(combinations)} combinations...\n")

for index, (process_std, measurement_std) in enumerate(
    combinations, start=1
):

    metrics = evaluate(
        train_ids,
        process_std,
        measurement_std
    )

    results.append({
        "process_std": process_std,
        "measurement_std": measurement_std,
        **metrics
    })

    print(
        f"[{index}/{len(combinations)}] "
        f"Process={process_std:4} "
        f"Measurement={measurement_std:3} "
        f"Mean={metrics['kalman_mean']:.2f} px"
    )

results.sort(key=lambda r: r["kalman_mean"])

best = results[0]


# ==========================================
# SAVE ALL TUNING RESULTS
# ==========================================

results_csv = OUTPUT_DIR / "tuning_results.csv"

with open(results_csv, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# EVALUATE HELD-OUT TEST TRACKS
# ==========================================

test_metrics = evaluate(
    test_ids,
    best["process_std"],
    best["measurement_std"]
)

original_metrics = evaluate(
    test_ids,
    80.0,
    8.0
)

baseline = test_metrics["baseline_mean"]

improvement = (
    (baseline - test_metrics["kalman_mean"])
    / baseline * 100
    if baseline > 0 else 0
)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n========== KALMAN TUNING REPORT ==========")

print(f"Combinations tested: {len(combinations)}")

print("\nBEST TRAINING PARAMETERS:")
print(f"Process acceleration STD: {best['process_std']}")
print(f"Measurement STD: {best['measurement_std']}")
print(f"Training mean error: {best['kalman_mean']:.2f} px")

print("\nHELD-OUT TEST RESULTS:")
print(
    f"Original Kalman mean: "
    f"{original_metrics['kalman_mean']:.2f} px"
)
print(
    f"Tuned Kalman mean: "
    f"{test_metrics['kalman_mean']:.2f} px"
)
print(
    f"Last-position baseline: "
    f"{baseline:.2f} px"
)
print(
    f"Tuned median error: "
    f"{test_metrics['kalman_median']:.2f} px"
)
print(
    f"Tuned 95th percentile: "
    f"{test_metrics['kalman_p95']:.2f} px"
)
print(f"Improvement vs baseline: {improvement:.2f}%")
print(f"Test predictions: {test_metrics['count']}")

print(f"\nResults saved: {results_csv}")
print("==========================================")
