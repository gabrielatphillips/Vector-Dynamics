
from pathlib import Path
from collections import defaultdict
import csv
import math
import random
import numpy as np

# ============================================
# VECTOR DYNAMICS - PHASE 11
# RADAR + CAMERA FUSION PROTOTYPE
# ============================================

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "vision" / "results" / "tracking_data.csv"
OUT = ROOT / "fusion" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = random.Random(SEED)

# Simulated camera calibration
IMAGE_WIDTH = 1920
CAMERA_HFOV_DEG = 70.0

# Radar simulation settings
RADAR_RANGE_NOISE_M = 1.0
RADAR_ANGLE_NOISE_DEG = 1.0
RADAR_SPEED_NOISE_MPS = 0.5
RADAR_DETECTION_PROBABILITY = 0.95

# Association settings
MAX_ANGLE_DIFFERENCE_DEG = 5.0

# Project requirements
FRONT_NOMINAL_RANGE_M = 121.92
REAR_MIN_RANGE_M = 4.572
REAR_MAX_RANGE_M = 60.96

# This first simulation uses front-range targets.
# It does NOT implement oncoming-lane classification.
MODE = "front"


# ============================================
# HELPERS
# ============================================

def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def camera_bearing(center_x):
    """
    Approximate pinhole-camera horizontal bearing.
    Assumes a centered, rectified image.
    """
    focal_px = (
        IMAGE_WIDTH / 2
        / math.tan(math.radians(CAMERA_HFOV_DEG / 2))
    )

    return math.degrees(
        math.atan(
            (center_x - IMAGE_WIDTH / 2) / focal_px
        )
    )


def load_camera_tracks():
    observations = []

    with open(INPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            observations.append({
                "frame": int(row["frame"]),
                "time_s": float(row["timestamp_s"]),
                "track_id": int(row["track_id"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "confidence": float(row["confidence"])
            })

    observations.sort(
        key=lambda r: (
            r["time_s"], r["frame"], r["track_id"]
        )
    )
    return observations


# ============================================
# SYNTHETIC RADAR TARGET MODEL
# ============================================

def create_target_parameters(observations):
    """
    Assign each camera track a synthetic range
    and radial velocity.

    These values are NOT estimated from video.
    """

    ids = sorted({
        obs["track_id"] for obs in observations
    })

    parameters = {}

    for track_id in ids:
        if MODE == "front":
            initial_range = rng.uniform(35.0, 115.0)
        else:
            initial_range = rng.uniform(
                REAR_MIN_RANGE_M + 5.0,
                REAR_MAX_RANGE_M - 5.0
            )

        # Positive = range increasing (receding)
        # Negative = range decreasing (approaching)
        radial_velocity = rng.uniform(-8.0, 4.0)

        first_time = min(
            obs["time_s"]
            for obs in observations
            if obs["track_id"] == track_id
        )

        parameters[track_id] = {
            "initial_range_m": initial_range,
            "radial_velocity_mps": radial_velocity,
            "first_time_s": first_time
        }

    return parameters


def simulate_radar(camera_obs, parameters):
    """
    Generate radar measurements for the current
    camera frame.

    Ground-truth association IDs are retained
    ONLY for evaluating the simulation.
    """

    measurements = []

    for obs in camera_obs:
        if rng.random() > RADAR_DETECTION_PROBABILITY:
            continue

        track_id = obs["track_id"]
        params = parameters[track_id]

        elapsed = (
            obs["time_s"] - params["first_time_s"]
        )

        true_range = (
            params["initial_range_m"]
            + params["radial_velocity_mps"] * elapsed
        )

        if MODE == "front":
            if not 0.0 < true_range <= FRONT_NOMINAL_RANGE_M:
                continue
        else:
            if not (
                REAR_MIN_RANGE_M
                <= true_range
                <= REAR_MAX_RANGE_M
            ):
                continue

        # Synthetic angle is constructed from
        # camera bearing plus independent noise.
        # This is a software test, not independent
        # physical sensor validation.
        true_bearing = camera_bearing(
            obs["center_x"]
        )

        measured_range = max(
            0.0,
            true_range
            + rng.gauss(0, RADAR_RANGE_NOISE_M)
        )

        measured_angle = (
            true_bearing
            + rng.gauss(0, RADAR_ANGLE_NOISE_DEG)
        )

        measured_velocity = (
            params["radial_velocity_mps"]
            + rng.gauss(0, RADAR_SPEED_NOISE_MPS)
        )

        measurements.append({
            "radar_id": len(measurements) + 1,
            "time_s": obs["time_s"],
            "range_m": measured_range,
            "angle_deg": measured_angle,
            "radial_velocity_mps": measured_velocity,
            "truth_track_id": track_id,
            "true_range_m": true_range,
            "true_velocity_mps": params["radial_velocity_mps"]
        })

    return measurements


# ============================================
# CAMERA-RADAR ASSOCIATION
# ============================================

def associate_targets(camera_obs, radar_obs):
    """
    One-to-one greedy nearest-angle association.

    This is a baseline algorithm, not a complete
    multi-target association solution.
    """

    candidates = []

    for ci, camera in enumerate(camera_obs):
        bearing = camera_bearing(camera["center_x"])

        for ri, radar in enumerate(radar_obs):
            difference = abs(
                bearing - radar["angle_deg"]
            )

            if difference <= MAX_ANGLE_DIFFERENCE_DEG:
                candidates.append(
                    (difference, ci, ri)
                )

    candidates.sort()

    used_camera = set()
    used_radar = set()
    matches = []

    for difference, ci, ri in candidates:
        if ci in used_camera or ri in used_radar:
            continue

        used_camera.add(ci)
        used_radar.add(ri)

        matches.append((
            camera_obs[ci],
            radar_obs[ri],
            difference
        ))

    return matches


# ============================================
# MAIN FUSION PIPELINE
# ============================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing camera tracking data: {INPUT}"
        )

    observations = load_camera_tracks()

    if not observations:
        raise ValueError("No camera observations found")

    parameters = create_target_parameters(
        observations
    )

    frames = defaultdict(list)

    for obs in observations:
        frames[(obs["frame"], obs["time_s"])].append(obs)

    radar_rows = []
    fusion_rows = []

    total_radar = 0
    correct_matches = 0
    incorrect_matches = 0
    total_matches = 0

    for (frame, time_s), cameras in sorted(
        frames.items(),
        key=lambda item: (
            item[0][1], item[0][0]
        )
    ):
        radar = simulate_radar(
            cameras, parameters
        )

        total_radar += len(radar)

        for target in radar:
            radar_rows.append({
                "frame": frame,
                **target
            })

        matches = associate_targets(
            cameras, radar
        )

        for camera, target, angle_error in matches:
            correct = (
                camera["track_id"]
                == target["truth_track_id"]
            )

            total_matches += 1

            if correct:
                correct_matches += 1
            else:
                incorrect_matches += 1

            closing_speed = max(
                0.0,
                -target["radial_velocity_mps"]
            )

            # Range / closing speed is an
            # illustrative radial TTC, not a
            # safe-passing or collision decision.
            if closing_speed > 0.1:
                radial_ttc = (
                    target["range_m"] / closing_speed
                )
            else:
                radial_ttc = ""

            fusion_rows.append({
                "frame": frame,
                "time_s": time_s,
                "camera_track_id": camera["track_id"],
                "radar_id": target["radar_id"],
                "mode": MODE,
                "range_m": round(
                    target["range_m"], 3
                ),
                "range_ft": round(
                    target["range_m"] * 3.28084, 2
                ),
                "radial_velocity_mps": round(
                    target["radial_velocity_mps"], 3
                ),
                "closing_speed_mph": round(
                    closing_speed * 2.23694, 2
                ),
                "angle_deg": round(
                    target["angle_deg"], 3
                ),
                "association_angle_error_deg": round(
                    angle_error, 3
                ),
                "camera_confidence": camera["confidence"],
                "radial_ttc_s": (
                    round(radial_ttc, 2)
                    if radial_ttc != "" else ""
                ),
                "correct_simulated_match": correct,
                "range_error_m": round(
                    target["range_m"]
                    - target["true_range_m"], 3
                ),
                "velocity_error_mps": round(
                    target["radial_velocity_mps"]
                    - target["true_velocity_mps"], 3
                )
            })

    radar_fields = [
        "frame", "radar_id", "time_s",
        "range_m", "angle_deg",
        "radial_velocity_mps",
        "truth_track_id", "true_range_m",
        "true_velocity_mps"
    ]

    fusion_fields = [
        "frame", "time_s",
        "camera_track_id", "radar_id",
        "mode", "range_m", "range_ft",
        "radial_velocity_mps",
        "closing_speed_mph", "angle_deg",
        "association_angle_error_deg",
        "camera_confidence", "radial_ttc_s",
        "correct_simulated_match",
        "range_error_m", "velocity_error_mps"
    ]

    save_csv(
        OUT / "simulated_radar.csv",
        radar_rows,
        radar_fields
    )

    save_csv(
        OUT / "fused_targets.csv",
        fusion_rows,
        fusion_fields
    )

    match_accuracy = (
        100 * correct_matches / total_matches
        if total_matches else 0
    )

    association_coverage = (
        100 * total_matches / total_radar
        if total_radar else 0
    )

    print("\n========== PHASE 11: RADAR FUSION ==========")
    print(f"Mode: {MODE}")
    print(f"Camera observations: {len(observations)}")
    print(f"Simulated radar detections: {total_radar}")
    print(f"Associated targets: {total_matches}")
    print(f"Correct associations: {correct_matches}")
    print(f"Incorrect associations: {incorrect_matches}")
    print(
        f"Association precision: "
        f"{match_accuracy:.2f}%"
    )
    print(
        f"Radar association coverage: "
        f"{association_coverage:.2f}%"
    )
    print("-------------------------------------------")
    print(f"Saved: {OUT / 'simulated_radar.csv'}")
    print(f"Saved: {OUT / 'fused_targets.csv'}")
    print("===========================================")


if __name__ == "__main__":
    main()
