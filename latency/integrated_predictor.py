
from pathlib import Path
from collections import defaultdict
import csv
import heapq
import math
import random

import numpy as np

# ==========================================
# VECTOR DYNAMICS
# INTEGRATED VEHICLE PREDICTION PROTOTYPE
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "vision" / "results" / "tracking_data.csv"
OUTPUT_DIR = ROOT / "latency" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "integrated_predictions.csv"
SUMMARY_CSV = OUTPUT_DIR / "integrated_summary.csv"

# Tuned image-space Kalman parameters
PROCESS_ACCEL_STD = 1280.0
MEASUREMENT_STD = 2.0

# Receiver timing and packet simulation
MAX_MEASUREMENT_AGE_S = 0.5
MAX_FILTER_GAP_S = 0.5

# Use identical video tracks in each condition
SCENARIOS = [
    ("normal", 40.0, 8.0),
    ("congested", 100.0, 25.0),
    ("recovered", 40.0, 8.0),
]

# Simulate losing every Nth packet.
# Set to 0 to disable packet loss.
DROP_EVERY_N = 0

# A fixed seed makes the experiment repeatable
RANDOM_SEED = 42

# Jacobson-Karels constants
ALPHA = 1.0 / 8.0
BETA = 1.0 / 4.0
K = 4.0
MIN_RTO_MS = 100.0


# ==========================================
# KALMAN FILTER
# ==========================================

class KalmanFilter:

    def __init__(self, x, y):

        self.state = np.array(
            [x, y, 0.0, 0.0], dtype=float
        )

        self.P = np.diag([
            100.0, 100.0, 10000.0, 10000.0
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.R = np.eye(2) * MEASUREMENT_STD**2

    @staticmethod
    def matrices(dt):

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

        return F, Q

    def predict(self, dt):

        if dt <= 0:
            return

        F, Q = self.matrices(dt)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, x, y):

        z = np.array([x, y], dtype=float)
        innovation = z - self.H @ self.state

        S = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(
            S, self.H @ self.P
        ).T

        self.state = self.state + gain @ innovation

        I = np.eye(4)
        A = I - gain @ self.H

        # Joseph covariance update
        self.P = A @ self.P @ A.T + gain @ self.R @ gain.T

    def estimate_at(self, dt):

        # Predict without modifying the saved filter state
        F, Q = self.matrices(max(dt, 0.0))

        predicted_state = F @ self.state
        predicted_cov = F @ self.P @ F.T + Q

        x = float(predicted_state[0])
        y = float(predicted_state[1])

        # Position covariance trace gives a simple
        # image-space uncertainty indicator.
        position_std_px = math.sqrt(
            max(
                0.0,
                predicted_cov[0, 0] +
                predicted_cov[1, 1]
            )
        )

        return x, y, position_std_px


# ==========================================
# JACOBSON-KARELS RTT ESTIMATOR
# ==========================================

class RTTEstimator:

    def __init__(self):
        self.srtt = None
        self.rttvar = None
        self.rto = None

    def update(self, measured_rtt_ms):

        if measured_rtt_ms <= 0:
            return

        if self.srtt is None:
            self.srtt = measured_rtt_ms
            self.rttvar = measured_rtt_ms / 2.0

        else:
            old_srtt = self.srtt

            self.rttvar = (
                (1.0 - BETA) * self.rttvar
                + BETA * abs(
                    old_srtt - measured_rtt_ms
                )
            )

            self.srtt = (
                (1.0 - ALPHA) * old_srtt
                + ALPHA * measured_rtt_ms
            )

        self.rto = max(
            MIN_RTO_MS,
            self.srtt + K * self.rttvar
        )


# ==========================================
# LOAD VEHICLE TRACKS
# ==========================================

def load_tracks():

    tracks = defaultdict(list)

    with open(
        INPUT_CSV, newline="", encoding="utf-8"
    ) as f:

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
        records.sort(
            key=lambda r: (r["time"], r["frame"])
        )

    return tracks


# ==========================================
# SIMULATE ONE NETWORK CONDITION
# ==========================================

def simulate_scenario(
    scenario_name,
    base_delay_ms,
    jitter_ms,
    tracks
):

    rng = random.Random(
        RANDOM_SEED + len(scenario_name)
    )

    # Each track gets an independent Kalman filter.
    filters = {}
    filter_times = {}
    last_capture_times = {}

    # Receiver packets ordered by ARRIVAL TIME.
    packets = []

    # Reference positions indexed by video frame
    reference_by_frame = defaultdict(list)

    all_frames = set()

    for track_id, records in tracks.items():

        for index, record in enumerate(records):

            frame = record["frame"]
            capture_time = record["time"]

            all_frames.add((frame, capture_time))

            reference_by_frame[frame].append(
                (track_id, record)
            )

            if (
                DROP_EVERY_N > 0
                and (index + 1) % DROP_EVERY_N == 0
            ):
                continue

            delay = max(
                1.0,
                rng.gauss(base_delay_ms, jitter_ms)
            )

            arrival_time = (
                capture_time + delay / 1000.0
            )

            heapq.heappush(
                packets,
                (
                    arrival_time,
                    track_id,
                    capture_time,
                    record["x"],
                    record["y"]
                )
            )

    # Synthetic RTT probes, generated independently
    # of vehicle measurement packets.
    rtt_estimator = RTTEstimator()

    rows = []
    received_packets = 0
    stale_count = 0
    unavailable_count = 0

    sorted_frames = sorted(all_frames)

    packet_index = 0

    for frame, current_time in sorted_frames:

        # --------------------------------------
        # 1. Receive packets available by NOW
        # --------------------------------------

        while (
            packet_index < len(packets)
            and packets[packet_index][0]
            <= current_time + 1e-9
        ):

            (
                arrival_time,
                track_id,
                capture_time,
                measured_x,
                measured_y
            ) = packets[packet_index]

            packet_index += 1
            received_packets += 1

            # Simulated round-trip probe.
            # This is a synthetic measurement, not
            # a real ESP32 RTT observation.
            return_delay_ms = max(
                1.0,
                rng.gauss(
                    base_delay_ms,
                    jitter_ms
                )
            )

            outbound_delay_ms = (
                arrival_time - capture_time
            ) * 1000.0

            simulated_rtt_ms = (
                outbound_delay_ms
                + return_delay_ms
            )

            rtt_estimator.update(
                simulated_rtt_ms
            )

            if track_id not in filters:

                filters[track_id] = KalmanFilter(
                    measured_x,
                    measured_y
                )

                filter_times[track_id] = capture_time
                last_capture_times[track_id] = capture_time

                continue

            dt = (
                capture_time - filter_times[track_id]
            )

            # Drop out-of-order measurements.
            # A later version can use buffered
            # out-of-sequence measurement updates.
            if dt <= 0:
                continue

            if dt > MAX_FILTER_GAP_S:

                filters[track_id] = KalmanFilter(
                    measured_x,
                    measured_y
                )

            else:

                filters[track_id].predict(dt)
                filters[track_id].update(
                    measured_x,
                    measured_y
                )

            filter_times[track_id] = capture_time
            last_capture_times[track_id] = capture_time

        # --------------------------------------
        # 2. Evaluate current vehicle positions
        # --------------------------------------

        for track_id, reference in reference_by_frame[frame]:

            if track_id not in filters:
                unavailable_count += 1
                continue

            measurement_age_s = (
                current_time
                - last_capture_times[track_id]
            )

            if measurement_age_s < -1e-9:
                raise RuntimeError(
                    "Receiver used a future measurement"
                )

            if (
                measurement_age_s
                > MAX_MEASUREMENT_AGE_S
            ):
                stale_count += 1
                continue

            predicted_x, predicted_y, uncertainty = (
                filters[track_id].estimate_at(
                    measurement_age_s
                )
            )

            error_px = math.hypot(
                reference["x"] - predicted_x,
                reference["y"] - predicted_y
            )

            # Prototype link-health classification.
            # These thresholds are NOT validated
            # vehicle-safety thresholds.
            if measurement_age_s >= 0.3:
                status = "STALE_WARNING"
            elif measurement_age_s >= 0.15:
                status = "CAUTION"
            else:
                status = "NORMAL"

            rows.append({
                "scenario": scenario_name,
                "frame": frame,
                "timestamp_s": round(
                    current_time, 4
                ),
                "track_id": track_id,
                "measurement_age_ms": round(
                    measurement_age_s * 1000.0, 2
                ),
                "predicted_x": round(
                    predicted_x, 3
                ),
                "predicted_y": round(
                    predicted_y, 3
                ),
                "reference_x": reference["x"],
                "reference_y": reference["y"],
                "error_px": round(error_px, 3),
                "uncertainty_px": round(
                    uncertainty, 3
                ),
                "srtt_ms": (
                    round(rtt_estimator.srtt, 3)
                    if rtt_estimator.srtt is not None
                    else ""
                ),
                "rttvar_ms": (
                    round(rtt_estimator.rttvar, 3)
                    if rtt_estimator.rttvar is not None
                    else ""
                ),
                "rto_ms": (
                    round(rtt_estimator.rto, 3)
                    if rtt_estimator.rto is not None
                    else ""
                ),
                "status": status
            })

    return (
        rows,
        received_packets,
        stale_count,
        unavailable_count
    )


# ==========================================
# RUN THE INTEGRATED EXPERIMENT
# ==========================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Tracking data not found: {INPUT_CSV}"
        )

    tracks = load_tracks()

    all_rows = []
    summaries = []

    for name, delay_ms, jitter_ms in SCENARIOS:

        (
            rows,
            received,
            stale,
            unavailable
        ) = simulate_scenario(
            name,
            delay_ms,
            jitter_ms,
            tracks
        )

        all_rows.extend(rows)

        errors = [
            r["error_px"] for r in rows
        ]

        if errors:
            mean_error = float(
                np.mean(errors)
            )
            p95_error = float(
                np.percentile(errors, 95)
            )
        else:
            mean_error = float("nan")
            p95_error = float("nan")

        summaries.append({
            "scenario": name,
            "mean_error_px": round(
                mean_error, 3
            ),
            "p95_error_px": round(
                p95_error, 3
            ),
            "evaluated_positions": len(rows),
            "stale_positions": stale,
            "unavailable_positions": unavailable,
            "received_packets": received
        })

    # Save detailed results
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # Save scenario summary
    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys())
        )

        writer.writeheader()
        writer.writerows(summaries)

    print(
        "\n========== INTEGRATED PREDICTOR =========="
    )

    print(f"Vehicle track IDs: {len(tracks)}")

    for s in summaries:

        print(
            f"{s['scenario'].capitalize():10} | "
            f"Mean={s['mean_error_px']:7.2f} px | "
            f"P95={s['p95_error_px']:7.2f} px | "
            f"Evaluated={s['evaluated_positions']:4} | "
            f"Stale={s['stale_positions']:4} | "
            f"Unavailable={s['unavailable_positions']:4}"
        )

    print("------------------------------------------")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print("==========================================")


if __name__ == "__main__":
    main()
