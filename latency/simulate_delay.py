
from pathlib import Path
from collections import defaultdict
import csv
import heapq

import numpy as np

# ==========================================
# VECTOR DYNAMICS
# RECEIVER-SIDE LATENCY SIMULATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "vision" / "results" / "tracking_data.csv"
OUTPUT_DIR = ROOT / "latency" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESS_ACCEL_STD = 1280.0
MEASUREMENT_STD = 2.0

DELAYS_MS = [0, 100, 200, 300]
DROP_FRAMES = [0, 3, 5, 10]

MAX_PREDICTION_SECONDS = 0.5
MAX_MEASUREMENT_GAP_SECONDS = 0.5


# ==========================================
# KALMAN FILTER
# ==========================================

class KalmanFilter:

    def __init__(self, x, y):

        self.state = np.array(
            [x, y, 0.0, 0.0],
            dtype=float
        )

        self.P = np.diag([
            100.0,
            100.0,
            10000.0,
            10000.0
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.R = np.eye(2) * MEASUREMENT_STD**2

    def predict(self, dt):

        if dt <= 0:
            return

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

        Q = PROCESS_ACCEL_STD**2 * (G @ G.T)

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

    def predict_position(self, dt):

        # Non-mutating forward prediction
        return np.array([
            self.state[0] + dt * self.state[2],
            self.state[1] + dt * self.state[3]
        ])


# ==========================================
# LOAD TRACKING DATA
# ==========================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Missing tracking data: {INPUT_CSV}"
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
    records.sort(key=lambda r: (r["time"], r["frame"]))


# ==========================================
# PROCESS ONE TRACK
# ==========================================

def simulate_track(records, delay_ms, drop_count):

    delay_seconds = delay_ms / 1000.0

    # Each packet includes:
    # arrival time, sequence, capture time, x, y
    packet_queue = []

    for index, record in enumerate(records):

        # Keep one measurement, then drop N
        if index % (drop_count + 1) != 0:
            continue

        arrival_time = record["time"] + delay_seconds

        heapq.heappush(
            packet_queue,
            (
                arrival_time,
                index,
                record["time"],
                record["x"],
                record["y"]
            )
        )

    kf = None
    filter_time = None
    last_capture_time = None
    last_arrival_time = None

    results = []
    packet_index = 0

    # Evaluate at actual video timestamps
    for record in records:

        current_time = record["time"]

        # Process ONLY packets that have arrived
        while (
            packet_index < len(packet_queue)
            and packet_queue[packet_index][0]
            <= current_time + 1e-9
        ):

            (
                arrival_time,
                sequence,
                capture_time,
                measured_x,
                measured_y
            ) = packet_queue[packet_index]

            packet_index += 1

            if kf is None:
                kf = KalmanFilter(measured_x, measured_y)
                filter_time = capture_time

            else:

                dt = capture_time - filter_time

                # Ignore stale/out-of-order captures
                if dt <= 0:
                    continue

                # Reset after long measurement gaps
                if dt > MAX_MEASUREMENT_GAP_SECONDS:
                    kf = KalmanFilter(measured_x, measured_y)

                else:
                    kf.predict(dt)
                    kf.update(measured_x, measured_y)

                filter_time = capture_time

            last_capture_time = capture_time
            last_arrival_time = arrival_time

        # No packet has arrived yet
        if kf is None:
            continue

        # Age of most recent available measurement
        age = current_time - last_capture_time

        if age < -1e-9:
            raise RuntimeError(
                "Future measurement entered receiver state"
            )

        if age > MAX_PREDICTION_SECONDS:
            continue

        # Predict from measurement time to NOW
        predicted = kf.predict_position(max(0.0, age))

        error = np.hypot(
            record["x"] - predicted[0],
            record["y"] - predicted[1]
        )

        results.append({
            "track_id": None,
            "frame": record["frame"],
            "evaluation_time_s": round(current_time, 4),
            "delay_ms": delay_ms,
            "drop_frames": drop_count,
            "last_capture_time_s": round(
                last_capture_time, 4
            ),
            "last_arrival_time_s": round(
                last_arrival_time, 4
            ),
            "measurement_age_ms": round(
                age * 1000, 2
            ),
            "predicted_x": round(predicted[0], 3),
            "predicted_y": round(predicted[1], 3),
            "reference_x": record["x"],
            "reference_y": record["y"],
            "error_px": round(error, 3)
        })

    return results


# ==========================================
# RUN ALL SCENARIOS
# ==========================================

all_scenarios = {}
eligible_keys = None

for delay_ms in DELAYS_MS:

    for drop_count in DROP_FRAMES:

        scenario_rows = []

        for track_id, records in tracks.items():

            rows = simulate_track(
                records,
                delay_ms,
                drop_count
            )

            for row in rows:
                row["track_id"] = track_id

            scenario_rows.extend(rows)

        key = (delay_ms, drop_count)

        all_scenarios[key] = scenario_rows

        current_keys = {
            (r["track_id"], r["frame"])
            for r in scenario_rows
        }

        if eligible_keys is None:
            eligible_keys = current_keys
        else:
            eligible_keys &= current_keys


# ==========================================
# COMPARE SAME TIMESTAMPS
# ==========================================

summary = []

for (delay_ms, drop_count), rows in all_scenarios.items():

    # Fair comparison: same vehicle/frame
    comparable = [
        row for row in rows
        if (row["track_id"], row["frame"]) in eligible_keys
    ]

    errors = np.array([
        row["error_px"] for row in comparable
    ])

    ages = np.array([
        row["measurement_age_ms"]
        for row in comparable
    ])

    if len(errors) == 0:
        raise RuntimeError(
            "No timestamps shared by all scenarios. "
            "Adjust prediction horizon or drop settings."
        )

    summary.append({
        "delay_ms": delay_ms,
        "drop_frames": drop_count,
        "mean_error_px": round(
            float(np.mean(errors)), 3
        ),
        "median_error_px": round(
            float(np.median(errors)), 3
        ),
        "p95_error_px": round(
            float(np.percentile(errors, 95)), 3
        ),
        "mean_age_ms": round(
            float(np.mean(ages)), 2
        ),
        "observations": len(errors)
    })

    output_file = (
        OUTPUT_DIR /
        f"receiver_delay_{delay_ms}_drop_{drop_count}.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=comparable[0].keys()
        )

        writer.writeheader()
        writer.writerows(comparable)

    print(
        f"Delay={delay_ms:3} ms | "
        f"Drops={drop_count:2} | "
        f"Mean={np.mean(errors):7.2f} px | "
        f"P95={np.percentile(errors, 95):7.2f} px | "
        f"Age={np.mean(ages):6.1f} ms"
    )


# ==========================================
# SAVE SUMMARY
# ==========================================

summary_file = OUTPUT_DIR / "receiver_delay_summary.csv"

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


print("\n========== RECEIVER LATENCY REPORT ==========")
print(f"Vehicle track IDs: {len(tracks)}")
print(f"Scenarios tested: {len(summary)}")
print(f"Common evaluation points: {len(eligible_keys)}")
print(f"Saved: {summary_file}")
print("=============================================")
