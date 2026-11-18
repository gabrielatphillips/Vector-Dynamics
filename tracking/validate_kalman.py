
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# VECTOR DYNAMICS - KALMAN VALIDATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "tracking" / "kalman_results.csv"
OUTPUT_DIR = ROOT / "tracking" / "validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# COMMAND LINE OPTIONS
# ==========================================

parser = argparse.ArgumentParser(
    description="Validate vehicle Kalman predictions"
)

parser.add_argument(
    "--track",
    type=int,
    default=1,
    help="Vehicle track ID to visualize"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=30.0,
    help="Large prediction error threshold in pixels"
)

args = parser.parse_args()


# ==========================================
# LOAD KALMAN RESULTS
# ==========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Kalman results not found: {INPUT_FILE}"
    )

tracks = defaultdict(list)

with open(
    INPUT_FILE,
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        track_id = int(row["track_id"])

        tracks[track_id].append({
            "frame": int(row["frame"]),
            "time": float(row["timestamp_s"]),
            "status": row["status"],
            "measured_x": float(row["measured_x"]),
            "measured_y": float(row["measured_y"]),
            "predicted_x": float(row["predicted_x"]),
            "predicted_y": float(row["predicted_y"]),
            "filtered_x": float(row["filtered_x"]),
            "filtered_y": float(row["filtered_y"]),
            "prediction_error": float(
                row["prediction_error_px"]
            )
        })

for records in tracks.values():
    records.sort(key=lambda r: (r["frame"], r["time"]))


# ==========================================
# CALCULATE BASELINE ERRORS
# ==========================================

all_kalman_errors = []
all_baseline_errors = []

summary = []
large_errors = []

for track_id, records in sorted(tracks.items()):

    kalman_errors = []
    baseline_errors = []

    previous = None

    for record in records:

        if record["status"] == "updated":

            if previous is not None:

                baseline_error = math.hypot(
                    record["measured_x"] -
                    previous["measured_x"],

                    record["measured_y"] -
                    previous["measured_y"]
                )

                kalman_error = record["prediction_error"]

                kalman_errors.append(kalman_error)
                baseline_errors.append(baseline_error)

                all_kalman_errors.append(kalman_error)
                all_baseline_errors.append(baseline_error)

                if kalman_error > args.threshold:
                    large_errors.append({
                        "track_id": track_id,
                        "frame": record["frame"],
                        "timestamp_s": record["time"],
                        "kalman_error_px": round(
                            kalman_error, 3
                        ),
                        "baseline_error_px": round(
                            baseline_error, 3
                        )
                    })

        previous = record

    if kalman_errors:

        summary.append({
            "track_id": track_id,
            "observations": len(records),
            "predictions": len(kalman_errors),
            "kalman_mean_px": round(
                np.mean(kalman_errors), 3
            ),
            "baseline_mean_px": round(
                np.mean(baseline_errors), 3
            ),
            "kalman_median_px": round(
                np.median(kalman_errors), 3
            ),
            "baseline_median_px": round(
                np.median(baseline_errors), 3
            )
        })


# ==========================================
# SAVE VALIDATION SUMMARY
# ==========================================

summary_file = OUTPUT_DIR / "validation_summary.csv"

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "track_id",
            "observations",
            "predictions",
            "kalman_mean_px",
            "baseline_mean_px",
            "kalman_median_px",
            "baseline_median_px"
        ]
    )

    writer.writeheader()
    writer.writerows(summary)


# ==========================================
# SAVE LARGE ERRORS
# ==========================================

large_error_file = OUTPUT_DIR / "large_errors.csv"

with open(
    large_error_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "track_id",
            "frame",
            "timestamp_s",
            "kalman_error_px",
            "baseline_error_px"
        ]
    )

    writer.writeheader()
    writer.writerows(large_errors)


# ==========================================
# GRAPH SELECTED VEHICLE TRACK
# ==========================================

if args.track not in tracks:
    raise ValueError(
        f"Track {args.track} does not exist. "
        f"Available IDs: {sorted(tracks.keys())}"
    )

records = tracks[args.track]

times = [r["time"] for r in records]

measured_x = [r["measured_x"] for r in records]
measured_y = [r["measured_y"] for r in records]

predicted_x = [r["predicted_x"] for r in records]
predicted_y = [r["predicted_y"] for r in records]

filtered_x = [r["filtered_x"] for r in records]
filtered_y = [r["filtered_y"] for r in records]

errors = [r["prediction_error"] for r in records]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 11),
    sharex=True
)

axes[0].plot(
    times, measured_x,
    label="Measured X",
    linewidth=1.5
)

axes[0].plot(
    times, predicted_x,
    label="Predicted X",
    linestyle="--"
)

axes[0].plot(
    times, filtered_x,
    label="Filtered X",
    linestyle=":"
)

axes[0].set_ylabel("X Position (pixels)")
axes[0].set_title(
    f"Vehicle Track {args.track} - X Position"
)
axes[0].legend()
axes[0].grid(True)


axes[1].plot(
    times, measured_y,
    label="Measured Y",
    linewidth=1.5
)

axes[1].plot(
    times, predicted_y,
    label="Predicted Y",
    linestyle="--"
)

axes[1].plot(
    times, filtered_y,
    label="Filtered Y",
    linestyle=":"
)

axes[1].set_ylabel("Y Position (pixels)")
axes[1].set_title(
    f"Vehicle Track {args.track} - Y Position"
)
axes[1].legend()
axes[1].grid(True)


axes[2].plot(
    times,
    errors,
    label="Prediction Error"
)

axes[2].axhline(
    args.threshold,
    linestyle="--",
    label="Error Threshold"
)

axes[2].set_ylabel("Error (pixels)")
axes[2].set_xlabel("Video Time (seconds)")
axes[2].set_title("One-Step Prediction Error")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

track_graph = (
    OUTPUT_DIR /
    f"track_{args.track}_validation.png"
)

plt.savefig(track_graph, dpi=150)
plt.close()


# ==========================================
# ERROR DISTRIBUTION HISTOGRAM
# ==========================================

if all_kalman_errors:

    plt.figure(figsize=(11, 6))

    plt.hist(
        all_kalman_errors,
        bins=50,
        alpha=0.6,
        label="Kalman"
    )

    plt.hist(
        all_baseline_errors,
        bins=50,
        alpha=0.6,
        label="Last Position Baseline"
    )

    plt.xlabel("Prediction Error (pixels)")
    plt.ylabel("Number of Predictions")
    plt.title("Kalman vs Baseline Error Distribution")
    plt.legend()
    plt.grid(True)

    histogram_file = (
        OUTPUT_DIR / "error_histogram.png"
    )

    plt.tight_layout()
    plt.savefig(histogram_file, dpi=150)
    plt.close()


# ==========================================
# PRINT FINAL REPORT
# ==========================================

print("\n========== KALMAN VALIDATION ==========")

print(f"Total track IDs: {len(tracks)}")
print(f"Predictions compared: {len(all_kalman_errors)}")

if all_kalman_errors:

    kalman_mean = np.mean(all_kalman_errors)
    baseline_mean = np.mean(all_baseline_errors)

    print(
        f"Kalman mean error: "
        f"{kalman_mean:.2f} px"
    )

    print(
        f"Baseline mean error: "
        f"{baseline_mean:.2f} px"
    )

    print(
        f"Kalman median error: "
        f"{np.median(all_kalman_errors):.2f} px"
    )

    print(
        f"Baseline median error: "
        f"{np.median(all_baseline_errors):.2f} px"
    )

    improvement = (
        (baseline_mean - kalman_mean)
        / baseline_mean * 100
        if baseline_mean > 0
        else 0
    )

    print(
        f"Improvement over baseline: "
        f"{improvement:.2f}%"
    )

print(
    f"Large prediction errors "
    f"(>{args.threshold} px): "
    f"{len(large_errors)}"
)

print(f"\nSelected track: {args.track}")
print(f"Track observations: {len(records)}")

print("\nOutput files:")
print(summary_file)
print(large_error_file)
print(track_graph)

print("=======================================")
