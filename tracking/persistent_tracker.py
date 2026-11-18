
from pathlib import Path
from collections import defaultdict
import csv
import heapq
import math
import random

import numpy as np

# ==========================================
# VECTOR DYNAMICS - PHASE 7
# PERSISTENT TRACKING + NETWORK RECOVERY
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "vision" / "results" / "tracking_data.csv"
OUT = ROOT / "tracking" / "persistent_results"
OUT.mkdir(parents=True, exist_ok=True)

PROCESS_STD = 1280.0
MEASUREMENT_STD = 2.0

# Prototype thresholds, NOT vehicle safety limits
MISSING_AFTER_S = 0.10
STALE_AFTER_S = 0.30
EXPIRE_AFTER_S = 0.50
RESET_GAP_S = 0.50

SEED = 42

# Jacobson-Karels parameters
ALPHA = 1 / 8
BETA = 1 / 4
MIN_RTO_MS = 100.0

# Simulation configurations
SCENARIOS = {
    "baseline": {
        "normal": (0.0, 0.0, 0),
        "congested": (0.0, 0.0, 0),
        "recovered": (0.0, 0.0, 0),
    },
    "network_test": {
        "normal": (40.0, 8.0, 0),
        "congested": (100.0, 25.0, 7),
        "recovered": (40.0, 8.0, 0),
    },
}


# ==========================================
# KALMAN FILTER
# ==========================================

class Kalman:

    def __init__(self, x, y):
        self.x = np.array([x, y, 0., 0.])
        self.P = np.diag([100., 100., 10000., 10000.])

        self.H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        self.R = np.eye(2) * MEASUREMENT_STD**2

    def matrices(self, dt):
        F = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        G = np.array([
            [0.5 * dt**2, 0.],
            [0., 0.5 * dt**2],
            [dt, 0.],
            [0., dt]
        ])
        Q = PROCESS_STD**2 * G @ G.T
        return F, Q

    def predict(self, dt):
        if dt <= 0:
            return
        F, Q = self.matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, x, y):
        z = np.array([x, y])
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T

        self.x += K @ residual
        A = np.eye(4) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T

    def estimate(self, dt):
        F, Q = self.matrices(max(dt, 0.))
        x = F @ self.x
        P = F @ self.P @ F.T + Q

        uncertainty = math.sqrt(
            max(0., P[0, 0] + P[1, 1])
        )
        return x[0], x[1], uncertainty


# ==========================================
# RTT ESTIMATOR
# ==========================================

class RTT:

    def __init__(self):
        self.srtt = None
        self.var = None
        self.rto = None

    def update(self, sample):
        if sample <= 0:
            return

        if self.srtt is None:
            self.srtt = sample
            self.var = sample / 2
        else:
            old = self.srtt
            self.var = (
                (1 - BETA) * self.var
                + BETA * abs(old - sample)
            )
            self.srtt = (
                (1 - ALPHA) * old
                + ALPHA * sample
            )

        self.rto = max(
            MIN_RTO_MS,
            self.srtt + 4 * self.var
        )


# ==========================================
# PERSISTENT VEHICLE TRACK
# ==========================================

class VehicleTrack:

    def __init__(self, x, y, capture_time, arrival_time):
        self.kf = Kalman(x, y)
        self.filter_time = capture_time
        self.last_capture = capture_time
        self.last_arrival = arrival_time
        self.received_count = 1

    def receive(self, x, y, capture_time, arrival_time):
        dt = capture_time - self.filter_time

        # Ignore duplicate or older captures
        if dt <= 0:
            return False

        if dt > RESET_GAP_S:
            self.kf = Kalman(x, y)
            self.received_count = 1
        else:
            self.kf.predict(dt)
            self.kf.update(x, y)
            self.received_count += 1

        self.filter_time = capture_time
        self.last_capture = capture_time
        self.last_arrival = arrival_time
        return True

    def status(self, now):
        age = now - self.last_capture

        if age > EXPIRE_AFTER_S:
            return "EXPIRED"
        if age > STALE_AFTER_S:
            return "STALE"
        if age > MISSING_AFTER_S:
            return "MISSING"
        if self.received_count == 1:
            return "NEW"
        return "TRACKING"

    def estimate(self, now):
        age = max(0., now - self.filter_time)
        return self.kf.estimate(age)


# ==========================================
# READ VIDEO TRACKS
# ==========================================

def load_data():
    observations = []
    with open(INPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            observations.append({
                "frame": int(row["frame"]),
                "time": float(row["timestamp_s"]),
                "id": int(row["track_id"]),
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
            })

    observations.sort(
        key=lambda r: (r["time"], r["frame"], r["id"])
    )
    return observations


def network_phase(t, start, end):
    fraction = (t - start) / max(end - start, 1e-9)
    if fraction < 1 / 3:
        return "normal"
    if fraction < 2 / 3:
        return "congested"
    return "recovered"


def draw_delay(rng, mean, jitter):
    if mean == 0 and jitter == 0:
        return 0.
    return max(1., rng.gauss(mean, jitter))


# ==========================================
# SIMULATE A CONTINUOUS NETWORK TIMELINE
# ==========================================

def run_scenario(name, settings, observations):
    rng = random.Random(SEED)

    start = observations[0]["time"]
    end = observations[-1]["time"]

    # Event queue:
    # (arrival time, order, event type, payload)
    events = []
    order = 0

    # Build vehicle measurement packets
    per_track_count = defaultdict(int)

    for obs in observations:
        phase = network_phase(obs["time"], start, end)
        mean, jitter, drop_every = settings[phase]

        per_track_count[obs["id"]] += 1

        if (
            drop_every > 0
            and per_track_count[obs["id"]] % drop_every == 0
        ):
            continue

        delay_ms = draw_delay(rng, mean, jitter)
        arrival = obs["time"] + delay_ms / 1000.

        heapq.heappush(
            events,
            (arrival, order, "vehicle", obs)
        )
        order += 1

    # Independent RTT probes every 100 ms.
    # Probes retain their network phase at SEND time.
    probe_time = start
    while probe_time <= end:
        phase = network_phase(probe_time, start, end)
        mean, jitter, _ = settings[phase]

        outbound = draw_delay(rng, mean, jitter)
        inbound = draw_delay(rng, mean, jitter)
        measured_rtt = outbound + inbound

        completion = probe_time + measured_rtt / 1000.

        heapq.heappush(
            events,
            (completion, order, "rtt", measured_rtt)
        )
        order += 1
        probe_time += 0.1

    # Evaluation timeline: all unique video frames
    frames = defaultdict(list)
    for obs in observations:
        frames[(obs["frame"], obs["time"])].append(obs)

    tracks = {}
    rtt = RTT()
    rows = []
    received_packets = 0
    received_probes = 0

    # Track state is maintained across all three phases.
    for (frame, now), references in sorted(
        frames.items(), key=lambda item: item[0][1]
    ):

        # Deliver only events completed by current time
        while events and events[0][0] <= now + 1e-9:
            arrival, _, kind, payload = heapq.heappop(events)

            if kind == "rtt":
                rtt.update(payload)
                received_probes += 1
                continue

            obs = payload
            track_id = obs["id"]
            received_packets += 1

            if track_id not in tracks:
                tracks[track_id] = VehicleTrack(
                    obs["x"], obs["y"],
                    obs["time"], arrival
                )
            else:
                tracks[track_id].receive(
                    obs["x"], obs["y"],
                    obs["time"], arrival
                )

        phase = network_phase(now, start, end)

        # Evaluate each reference vehicle position
        for ref in references:
            track = tracks.get(ref["id"])

            if track is None:
                status = "UNAVAILABLE"
                age_ms = ""
                px = py = uncertainty = ""
                error = ""
            else:
                status = track.status(now)
                age_ms = round(
                    (now - track.last_capture) * 1000, 3
                )

                if status == "EXPIRED":
                    px = py = uncertainty = ""
                    error = ""
                else:
                    px, py, uncertainty = track.estimate(now)
                    error = math.hypot(
                        ref["x"] - px,
                        ref["y"] - py
                    )

            rows.append({
                "scenario": name,
                "phase": phase,
                "frame": frame,
                "time_s": round(now, 5),
                "track_id": ref["id"],
                "status": status,
                "age_ms": age_ms,
                "predicted_x": (
                    round(px, 3) if px != "" else ""
                ),
                "predicted_y": (
                    round(py, 3) if py != "" else ""
                ),
                "reference_x": ref["x"],
                "reference_y": ref["y"],
                "error_px": (
                    round(error, 3) if error != "" else ""
                ),
                "uncertainty_px": (
                    round(uncertainty, 3)
                    if uncertainty != "" else ""
                ),
                "srtt_ms": (
                    round(rtt.srtt, 3)
                    if rtt.srtt is not None else ""
                ),
                "rttvar_ms": (
                    round(rtt.var, 3)
                    if rtt.var is not None else ""
                ),
                "rto_ms": (
                    round(rtt.rto, 3)
                    if rtt.rto is not None else ""
                ),
            })

    return rows, received_packets, received_probes


# ==========================================
# REPORTING
# ==========================================

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(name, rows, packets, probes):
    summary = []

    for phase in ("normal", "congested", "recovered"):
        group = [r for r in rows if r["phase"] == phase]
        valid = [r for r in group if r["error_px"] != ""]
        errors = [float(r["error_px"]) for r in valid]

        summary.append({
            "scenario": name,
            "phase": phase,
            "reference_positions": len(group),
            "available_positions": len(valid),
            "availability_pct": round(
                100 * len(valid) / len(group), 2
            ),
            "unavailable": sum(
                r["status"] == "UNAVAILABLE" for r in group
            ),
            "expired": sum(
                r["status"] == "EXPIRED" for r in group
            ),
            "missing": sum(
                r["status"] == "MISSING" for r in group
            ),
            "stale": sum(
                r["status"] == "STALE" for r in group
            ),
            "mean_error_px": (
                round(float(np.mean(errors)), 3)
                if errors else ""
            ),
            "p95_error_px": (
                round(float(np.percentile(errors, 95)), 3)
                if errors else ""
            ),
            "received_packets_total": packets,
            "completed_rtt_probes_total": probes
        })

    return summary


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find tracking CSV: {INPUT}"
        )

    observations = load_data()

    if not observations:
        raise ValueError("Tracking CSV is empty")

    all_summaries = []
    scenario_rows = {}

    for name, settings in SCENARIOS.items():
        rows, packets, probes = run_scenario(
            name, settings, observations
        )
        scenario_rows[name] = rows

        save_csv(OUT / f"{name}_tracks.csv", rows)

        all_summaries.extend(
            summarize(name, rows, packets, probes)
        )

    # Fair accuracy comparison:
    # only vehicle/frame positions valid in BOTH runs.
    baseline = {
        (r["track_id"], r["frame"]): r
        for r in scenario_rows["baseline"]
        if r["error_px"] != ""
    }
    network = {
        (r["track_id"], r["frame"]): r
        for r in scenario_rows["network_test"]
        if r["error_px"] != ""
    }

    common = sorted(set(baseline) & set(network))
    comparison = []

    for key in common:
        a, b = baseline[key], network[key]
        comparison.append({
            "track_id": key[0],
            "frame": key[1],
            "phase": b["phase"],
            "baseline_error_px": a["error_px"],
            "network_error_px": b["error_px"],
            "network_status": b["status"],
            "network_age_ms": b["age_ms"]
        })

    save_csv(OUT / "common_comparison.csv", comparison)
    save_csv(OUT / "persistent_summary.csv", all_summaries)

    print("\n========== PERSISTENT TRACKING REPORT ==========")
    print(f"Total observations: {len(observations)}")
    print(f"Common comparison points: {len(common)}")
    print()

    for s in all_summaries:
        print(
            f"{s['scenario']:12} | "
            f"{s['phase']:10} | "
            f"Coverage={s['availability_pct']:6.2f}% | "
            f"Mean={str(s['mean_error_px']):>7} px | "
            f"P95={str(s['p95_error_px']):>7} px | "
            f"Missing={s['missing']:3} | "
            f"Stale={s['stale']:3}"
        )

    print("-----------------------------------------------")
    print(f"Saved results: {OUT}")
    print("================================================")


if __name__ == "__main__":
    main()
