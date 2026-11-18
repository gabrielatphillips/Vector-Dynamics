
from pathlib import Path
from collections import defaultdict
import csv
import heapq
import math
import random

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# VECTOR DYNAMICS - PHASE 9
# CV vs CA MOTION MODEL COMPARISON
# FULL 2D COVARIANCE VALIDATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "vision" / "results" / "tracking_data.csv"
OUTPUT = ROOT / "tracking" / "motion_model_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42

MEASUREMENT_STD = 2.0

# Existing tuned CV process noise
CV_ACCEL_STD = 1280.0

# Initial CA process noise: acceleration-change
# (jerk) standard deviation, pixels/s^3.
# This is a starting assumption, not a tuned value.
CA_JERK_STD = 5000.0

MAX_AGE_S = 0.50
RESET_GAP_S = 0.50

CHI2_95 = 5.991464547

PHASES = ("normal", "congested", "recovered")

NETWORK = {
    "normal": (40.0, 8.0, 0),
    "congested": (100.0, 25.0, 7),
    "recovered": (40.0, 8.0, 0)
}


# ============================================
# CONSTANT VELOCITY MODEL
# State: x, y, vx, vy
# ============================================

class ConstantVelocity:

    def __init__(self, x, y):
        self.x = np.array(
            [x, y, 0.0, 0.0], dtype=float
        )

        self.P = np.diag([
            100.0, 100.0,
            10000.0, 10000.0
        ])

        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.R = np.eye(2) * MEASUREMENT_STD**2

    def matrices(self, dt):
        F = np.eye(4)

        F[0, 2] = dt
        F[1, 3] = dt

        G = np.array([
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt]
        ])

        Q = CV_ACCEL_STD**2 * G @ G.T

        return F, Q


# ============================================
# CONSTANT ACCELERATION MODEL
# State: x, y, vx, vy, ax, ay
# ============================================

class ConstantAcceleration:

    def __init__(self, x, y):
        self.x = np.array([
            x, y, 0.0, 0.0, 0.0, 0.0
        ], dtype=float)

        self.P = np.diag([
            100.0, 100.0,
            10000.0, 10000.0,
            1000000.0, 1000000.0
        ])

        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.R = np.eye(2) * MEASUREMENT_STD**2

    def matrices(self, dt):
        F = np.eye(6)

        F[0, 2] = dt
        F[1, 3] = dt

        F[0, 4] = 0.5 * dt**2
        F[1, 5] = 0.5 * dt**2

        F[2, 4] = dt
        F[3, 5] = dt

        # Discrete white-jerk approximation
        G = np.array([
            [dt**3 / 6.0, 0.0],
            [0.0, dt**3 / 6.0],
            [dt**2 / 2.0, 0.0],
            [0.0, dt**2 / 2.0],
            [dt, 0.0],
            [0.0, dt]
        ])

        Q = CA_JERK_STD**2 * G @ G.T

        return F, Q


# ============================================
# SHARED KALMAN OPERATIONS
# ============================================

def predict_in_place(model, dt):
    if dt <= 0:
        return

    F, Q = model.matrices(dt)

    model.x = F @ model.x
    model.P = F @ model.P @ F.T + Q

    model.P = (model.P + model.P.T) / 2


def measurement_update(model, x, y):
    z = np.array([x, y])

    innovation = z - model.H @ model.x

    S = model.H @ model.P @ model.H.T + model.R

    K = np.linalg.solve(
        S, model.H @ model.P
    ).T

    model.x = model.x + K @ innovation

    I = np.eye(len(model.x))
    A = I - K @ model.H

    # Joseph covariance update
    model.P = (
        A @ model.P @ A.T
        + K @ model.R @ K.T
    )

    model.P = (model.P + model.P.T) / 2


def estimate_at(model, dt):
    dt = max(0.0, dt)

    F, Q = model.matrices(dt)

    state = F @ model.x
    covariance = F @ model.P @ F.T + Q

    covariance = (
        covariance + covariance.T
    ) / 2

    return state, covariance


# ============================================
# FULL COVARIANCE VALIDATION
# ============================================

def covariance_metrics(state, P, true_x, true_y):
    predicted = state[:2]
    actual = np.array([true_x, true_y])

    error = actual - predicted

    euclidean = float(np.linalg.norm(error))

    # Position covariance: full 2x2 matrix
    Pxy = P[:2, :2]

    # Numerical stabilization only
    Pxy = (Pxy + Pxy.T) / 2

    eigenvalues = np.linalg.eigvalsh(Pxy)

    if eigenvalues[0] <= 0:
        raise ValueError(
            "Position covariance is not positive definite"
        )

    # Squared Mahalanobis distance
    d2 = float(
        error @ np.linalg.solve(Pxy, error)
    )

    # Ellipse axes at nominal 95% confidence
    axes = np.sqrt(CHI2_95 * eigenvalues)

    return {
        "error_px": euclidean,
        "mahalanobis_d2": d2,
        "inside95": d2 <= CHI2_95,
        "Pxx": float(Pxy[0, 0]),
        "Pxy": float(Pxy[0, 1]),
        "Pyy": float(Pxy[1, 1]),
        "ellipse_minor_px": float(axes[0]),
        "ellipse_major_px": float(axes[1])
    }


# ============================================
# DATA LOADING
# ============================================

def load_observations():
    rows = []

    with open(INPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame": int(row["frame"]),
                "time": float(row["timestamp_s"]),
                "id": int(row["track_id"]),
                "x": float(row["center_x"]),
                "y": float(row["center_y"])
            })

    rows.sort(
        key=lambda r: (
            r["time"], r["frame"], r["id"]
        )
    )

    return rows


def phase_for_time(t, start, end):
    fraction = (
        (t - start) / max(end - start, 1e-9)
    )

    if fraction < 1 / 3:
        return "normal"
    if fraction < 2 / 3:
        return "congested"

    return "recovered"


def create_network_events(observations):
    rng = random.Random(SEED)

    start = observations[0]["time"]
    end = observations[-1]["time"]

    counters = defaultdict(int)
    events = []

    for order, obs in enumerate(observations):
        phase = phase_for_time(
            obs["time"], start, end
        )

        mean, jitter, drop_every = NETWORK[phase]

        counters[obs["id"]] += 1

        if (
            drop_every
            and counters[obs["id"]] % drop_every == 0
        ):
            continue

        delay_ms = max(
            1.0, rng.gauss(mean, jitter)
        )

        arrival = obs["time"] + delay_ms / 1000

        heapq.heappush(
            events, (arrival, order, obs)
        )

    return events


# ============================================
# SIMULATE BOTH MODELS WITH IDENTICAL PACKETS
# ============================================

def run_model(model_class, model_name,
              observations, packet_events):

    events = list(packet_events)
    heapq.heapify(events)

    start = observations[0]["time"]
    end = observations[-1]["time"]

    frames = defaultdict(list)

    for obs in observations:
        frames[(obs["frame"], obs["time"])].append(obs)

    tracks = {}
    output = []

    for (frame, now), references in sorted(
        frames.items(),
        key=lambda item: (
            item[0][1], item[0][0]
        )
    ):

        # Deliver packets in arrival-time order.
        # No future measurements are accessible.
        while events and events[0][0] <= now + 1e-9:
            arrival, _, packet = heapq.heappop(events)

            tid = packet["id"]
            capture = packet["time"]

            if tid not in tracks:
                tracks[tid] = {
                    "filter": model_class(
                        packet["x"], packet["y"]
                    ),
                    "time": capture
                }
                continue

            track = tracks[tid]

            dt = capture - track["time"]

            # Ignore out-of-order older captures
            if dt <= 0:
                continue

            if dt > RESET_GAP_S:
                track["filter"] = model_class(
                    packet["x"], packet["y"]
                )
            else:
                predict_in_place(track["filter"], dt)

                measurement_update(
                    track["filter"],
                    packet["x"],
                    packet["y"]
                )

            track["time"] = capture

        phase = phase_for_time(now, start, end)

        for ref in references:
            tid = ref["id"]

            result = {
                "model": model_name,
                "phase": phase,
                "frame": frame,
                "time_s": now,
                "track_id": tid,
                "available": False,
                "age_ms": "",
                "error_px": "",
                "mahalanobis_d2": "",
                "inside95": "",
                "Pxx": "",
                "Pxy": "",
                "Pyy": "",
                "ellipse_minor_px": "",
                "ellipse_major_px": ""
            }

            if tid not in tracks:
                output.append(result)
                continue

            track = tracks[tid]

            age = now - track["time"]

            result["age_ms"] = round(
                age * 1000, 3
            )

            if age > MAX_AGE_S:
                output.append(result)
                continue

            state, P = estimate_at(
                track["filter"], age
            )

            metrics = covariance_metrics(
                state, P, ref["x"], ref["y"]
            )

            result.update(metrics)
            result["available"] = True

            output.append(result)

    return output


# ============================================
# STATISTICS
# ============================================

def summary(rows, model, phase):
    group = [
        r for r in rows
        if r["model"] == model
        and (
            phase == "overall"
            or r["phase"] == phase
        )
    ]

    valid = [
        r for r in group if r["available"]
    ]

    errors = [
        r["error_px"] for r in valid
    ]

    inside = [
        r["inside95"] for r in valid
    ]

    return {
        "model": model,
        "phase": phase,
        "total": len(group),
        "available": len(valid),
        "coverage_pct": (
            100 * len(valid) / len(group)
            if group else 0
        ),
        "mean_error_px": (
            float(np.mean(errors))
            if errors else float("nan")
        ),
        "p95_error_px": (
            float(np.percentile(errors, 95))
            if errors else float("nan")
        ),
        "inside95_pct": (
            100 * float(np.mean(inside))
            if inside else float("nan")
        )
    }


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================
# MAIN
# ============================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    observations = load_observations()

    if not observations:
        raise ValueError("No observations found")

    unique_ids = sorted({
        obs["id"] for obs in observations
    })

    # Deterministic held-out track-ID split.
    # No parameter tuning occurs on these IDs.
    rng = random.Random(SEED)
    rng.shuffle(unique_ids)

    split = int(0.7 * len(unique_ids))

    train_ids = set(unique_ids[:split])
    test_ids = set(unique_ids[split:])

    print("\n========== PHASE 9 ==========")
    print(f"Total observations: {len(observations)}")
    print(f"Training IDs reserved: {len(train_ids)}")
    print(f"Held-out test IDs: {len(test_ids)}")

    # Same simulated packet arrivals for both models
    events = create_network_events(observations)

    cv_rows = run_model(
        ConstantVelocity,
        "CV",
        observations,
        events
    )

    ca_rows = run_model(
        ConstantAcceleration,
        "CA",
        observations,
        events
    )

    all_rows = cv_rows + ca_rows

    save_csv(
        OUTPUT / "all_model_predictions.csv",
        all_rows
    )

    # Report held-out IDs only.
    test_rows = [
        r for r in all_rows
        if r["track_id"] in test_ids
    ]

    summaries = []

    for model in ("CV", "CA"):
        for phase in (
            "overall", *PHASES
        ):
            summaries.append(
                summary(test_rows, model, phase)
            )

    save_csv(
        OUTPUT / "heldout_summary.csv",
        summaries
    )

    # Matched available predictions on held-out IDs
    cv = {
        (r["frame"], r["track_id"]): r
        for r in test_rows
        if r["model"] == "CV"
        and r["available"]
    }

    ca = {
        (r["frame"], r["track_id"]): r
        for r in test_rows
        if r["model"] == "CA"
        and r["available"]
    }

    common = sorted(set(cv) & set(ca))

    matched = []

    for key in common:
        a = cv[key]
        b = ca[key]

        matched.append({
            "frame": key[0],
            "track_id": key[1],
            "phase": a["phase"],
            "cv_error_px": a["error_px"],
            "ca_error_px": b["error_px"],
            "cv_inside95": a["inside95"],
            "ca_inside95": b["inside95"],
            "cv_d2": a["mahalanobis_d2"],
            "ca_d2": b["mahalanobis_d2"]
        })

    save_csv(
        OUTPUT / "matched_comparison.csv",
        matched
    )

    print(f"Matched test positions: {len(matched)}")
    print()

    for s in summaries:
        print(
            f"{s['model']:2} | "
            f"{s['phase']:10} | "
            f"Coverage={s['coverage_pct']:6.2f}% | "
            f"Mean={s['mean_error_px']:7.2f} px | "
            f"P95={s['p95_error_px']:7.2f} px | "
            f"Inside95={s['inside95_pct']:6.2f}%"
        )

    if matched:
        cv_error = np.array([
            r["cv_error_px"] for r in matched
        ])
        ca_error = np.array([
            r["ca_error_px"] for r in matched
        ])

        improvement = (
            100 * (
                np.mean(cv_error) - np.mean(ca_error)
            ) / np.mean(cv_error)
            if np.mean(cv_error) > 0 else 0
        )

        print("---------------------------------------")
        print(
            f"Matched CV mean: {np.mean(cv_error):.3f} px"
        )
        print(
            f"Matched CA mean: {np.mean(ca_error):.3f} px"
        )
        print(
            f"CA improvement: {improvement:+.2f}%"
        )

        # Error comparison plot
        plt.figure(figsize=(9, 5))

        plt.boxplot(
            [cv_error, ca_error],
            tick_labels=["CV", "CA"],
            showfliers=False
        )

        plt.ylabel("Prediction error (pixels)")
        plt.title(
            "Vector Dynamics: Held-Out Model Comparison"
        )
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT / "model_error_comparison.png",
            dpi=200
        )
        plt.close()

        # Proper full-covariance calibration
        plt.figure(figsize=(9, 5))

        rates = [
            100 * np.mean([
                r["cv_inside95"] for r in matched
            ]),
            100 * np.mean([
                r["ca_inside95"] for r in matched
            ])
        ]

        plt.bar(["CV", "CA"], rates)

        plt.axhline(
            95,
            linestyle="--",
            label="Nominal 95%"
        )

        plt.ylabel("Inside 95% ellipse (%)")
        plt.ylim(0, 105)
        plt.title("Full-Covariance Calibration")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT / "covariance_calibration.png",
            dpi=200
        )
        plt.close()

    print("---------------------------------------")
    print(f"Saved results: {OUTPUT}")
    print("=======================================")


if __name__ == "__main__":
    main()
