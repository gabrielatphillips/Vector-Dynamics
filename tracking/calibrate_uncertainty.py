
from pathlib import Path
from collections import defaultdict
import csv
import math
import random

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# VECTOR DYNAMICS - PHASE 10
# KALMAN COVARIANCE CALIBRATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "tracking"
    / "motion_model_results"
    / "all_model_predictions.csv"
)

OUTPUT = (
    ROOT / "tracking"
    / "calibration_results"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_COVERAGE = 0.95
CHI2_95 = 5.991464547

# At least this many training predictions
# before calibrating an individual age group.
MIN_BIN_SAMPLES = 50

PHASES = ("normal", "congested", "recovered")

AGE_BINS = (
    ("0-100 ms", 0.0, 100.0),
    ("100-200 ms", 100.0, 200.0),
    ("200-300 ms", 200.0, 300.0),
    ("300-400 ms", 300.0, 400.0),
    ("400-500 ms", 400.0, float("inf"))
)


# ============================================
# FILE HELPERS
# ============================================


def save_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"No rows to save: {path.name}")
        return

    # Collect every column from every row,
    # including fields absent from the first row.
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        ))

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)



def age_group(age):
    for label, lower, upper in AGE_BINS:
        if lower <= age < upper:
            return label
    return "unknown"


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing Phase 9 output: {INPUT}"
        )

    records = []

    with open(
        INPUT, newline="", encoding="utf-8"
    ) as f:

        for row in csv.DictReader(f):

            # Only calibrate the CV model.
            if row["model"] != "CV":
                continue

            valid = str(
                row["available"]
            ).strip().lower() == "true"

            record = {
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "phase": row["phase"],
                "available": valid,
            }

            if valid:
                record.update({
                    "age_ms": float(row["age_ms"]),
                    "error_px": float(
                        row["error_px"]
                    ),
                    "d2": float(
                        row["mahalanobis_d2"]
                    ),
                    "Pxx": float(row["Pxx"]),
                    "Pxy": float(row["Pxy"]),
                    "Pyy": float(row["Pyy"])
                })

                record["age_group"] = age_group(
                    record["age_ms"]
                )

            records.append(record)

    return records


# ============================================
# RECREATE PHASE 9 TRAIN / TEST SPLIT
# ============================================

def split_tracks(records):
    ids = sorted({
        r["track_id"] for r in records
    })

    rng = random.Random(SEED)
    rng.shuffle(ids)

    split = int(0.7 * len(ids))

    train_ids = set(ids[:split])
    test_ids = set(ids[split:])

    return train_ids, test_ids


# ============================================
# EMPIRICAL COVARIANCE CALIBRATION
# ============================================

def empirical_quantile(values, probability):
    """
    Conservative order-statistic quantile.

    For n samples, use rank:
        ceil((n + 1) * probability)

    This is a split-conformal-style
    calibration threshold for scalar scores.

    Its formal coverage guarantee requires
    exchangeability, which correlated video
    observations do not generally satisfy.
    """
    values = sorted(
        float(v) for v in values
        if np.isfinite(v)
    )

    if not values:
        raise ValueError(
            "Cannot calibrate with no valid scores"
        )

    rank = math.ceil(
        (len(values) + 1) * probability
    )

    rank = min(rank, len(values))

    return values[rank - 1]


def covariance_scale(records):
    """
    Original 95% ellipse:
        d2 <= CHI2_95

    If covariance is multiplied by k:
        new d2 = old d2 / k

    To achieve the empirical target:
        k = quantile(d2) / CHI2_95
    """
    scores = [
        r["d2"] for r in records
        if r["available"]
        and np.isfinite(r["d2"])
    ]

    if not scores:
        raise ValueError(
            "No valid calibration predictions"
        )

    threshold = empirical_quantile(
        scores, TARGET_COVERAGE
    )

    # Do not shrink covariance in this phase.
    return max(1.0, threshold / CHI2_95)


def build_calibration(train):
    valid = [
        r for r in train if r["available"]
    ]

    global_scale = covariance_scale(valid)

    age_scales = {}
    sample_counts = {}

    for label, _, _ in AGE_BINS:
        group = [
            r for r in valid
            if r["age_group"] == label
        ]

        sample_counts[label] = len(group)

        if len(group) >= MIN_BIN_SAMPLES:
            age_scales[label] = max(
                global_scale,
                covariance_scale(group)
            )
        else:
            age_scales[label] = global_scale

    return (
        global_scale,
        age_scales,
        sample_counts
    )


# ============================================
# APPLY CALIBRATION
# ============================================

def apply_calibration(
    records, global_scale, age_scales
):
    results = []

    for record in records:
        result = dict(record)

        if not record["available"]:
            result.update({
                "global_scale": "",
                "age_scale": "",
                "global_d2": "",
                "age_d2": "",
                "original_inside95": "",
                "global_inside95": "",
                "age_inside95": "",
                "calibrated_Pxx": "",
                "calibrated_Pxy": "",
                "calibrated_Pyy": ""
            })

            results.append(result)
            continue

        scale = age_scales[
            record["age_group"]
        ]

        original_d2 = record["d2"]

        global_d2 = (
            original_d2 / global_scale
        )

        age_d2 = original_d2 / scale

        result.update({
            "global_scale": global_scale,
            "age_scale": scale,
            "global_d2": global_d2,
            "age_d2": age_d2,
            "original_inside95": (
                original_d2 <= CHI2_95
            ),
            "global_inside95": (
                global_d2 <= CHI2_95
            ),
            "age_inside95": (
                age_d2 <= CHI2_95
            ),
            "calibrated_Pxx": (
                record["Pxx"] * scale
            ),
            "calibrated_Pxy": (
                record["Pxy"] * scale
            ),
            "calibrated_Pyy": (
                record["Pyy"] * scale
            )
        })

        results.append(result)

    return results


# ============================================
# EVALUATION
# ============================================

def summarize(records, split_name, group_name):
    valid = [
        r for r in records if r["available"]
    ]

    total = len(records)

    result = {
        "split": split_name,
        "group": group_name,
        "total_positions": total,
        "available_positions": len(valid),
        "coverage_pct": (
            100 * len(valid) / total
            if total else float("nan")
        )
    }

    if not valid:
        result.update({
            "mean_error_px": float("nan"),
            "p95_error_px": float("nan"),
            "original_inside95_pct": float("nan"),
            "global_inside95_pct": float("nan"),
            "age_inside95_pct": float("nan"),
            "mean_age_ms": float("nan")
        })
        return result

    errors = [
        r["error_px"] for r in valid
    ]

    result.update({
        "mean_error_px": float(
            np.mean(errors)
        ),
        "p95_error_px": float(
            np.percentile(errors, 95)
        ),
        "original_inside95_pct": (
            100 * np.mean([
                r["original_inside95"]
                for r in valid
            ])
        ),
        "global_inside95_pct": (
            100 * np.mean([
                r["global_inside95"]
                for r in valid
            ])
        ),
        "age_inside95_pct": (
            100 * np.mean([
                r["age_inside95"]
                for r in valid
            ])
        ),
        "mean_age_ms": float(
            np.mean([
                r["age_ms"] for r in valid
            ])
        )
    })

    return result


def evaluate_split(records, split_name):
    summaries = [
        summarize(
            records, split_name, "overall"
        )
    ]

    for phase in PHASES:
        group = [
            r for r in records
            if r["phase"] == phase
        ]

        summaries.append(
            summarize(
                group, split_name, phase
            )
        )

    for label, _, _ in AGE_BINS:
        group = [
            r for r in records
            if r["available"]
            and r["age_group"] == label
        ]

        if group:
            summaries.append(
                summarize(
                    group, split_name, label
                )
            )

    return summaries


# ============================================
# VISUALIZATION
# ============================================

def plot_coverage(summaries):
    rows = [
        r for r in summaries
        if r["split"] == "test"
        and r["group"] in PHASES
    ]

    labels = [
        r["group"].capitalize()
        for r in rows
    ]

    original = [
        r["original_inside95_pct"]
        for r in rows
    ]

    global_values = [
        r["global_inside95_pct"]
        for r in rows
    ]

    age_values = [
        r["age_inside95_pct"]
        for r in rows
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width,
        original,
        width,
        label="Original"
    )

    plt.bar(
        x,
        global_values,
        width,
        label="Global calibration"
    )

    plt.bar(
        x + width,
        age_values,
        width,
        label="Age calibration"
    )

    plt.axhline(
        95,
        linestyle="--",
        label="Nominal 95%"
    )

    plt.xticks(x, labels)
    plt.ylabel("Inside 95% ellipse (%)")
    plt.ylim(0, 105)
    plt.title(
        "Vector Dynamics: Held-Out Calibration"
    )
    plt.legend()
    plt.grid(
        True, axis="y", alpha=0.3
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "heldout_calibration.png",
        dpi=200
    )
    plt.close()


def plot_age(summaries):
    rows = [
        r for r in summaries
        if r["split"] == "test"
        and "ms" in r["group"]
    ]

    if not rows:
        return

    labels = [
        r["group"] for r in rows
    ]

    original = [
        r["original_inside95_pct"]
        for r in rows
    ]

    calibrated = [
        r["age_inside95_pct"]
        for r in rows
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width / 2,
        original,
        width,
        label="Original"
    )

    plt.bar(
        x + width / 2,
        calibrated,
        width,
        label="Age calibrated"
    )

    plt.axhline(
        95,
        linestyle="--",
        label="Nominal 95%"
    )

    plt.xticks(
        x, labels, rotation=20
    )

    plt.ylabel("Inside 95% ellipse (%)")
    plt.ylim(0, 105)
    plt.title(
        "Held-Out Calibration by Measurement Age"
    )
    plt.legend()
    plt.grid(
        True, axis="y", alpha=0.3
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "calibration_by_age.png",
        dpi=200
    )
    plt.close()


# ============================================
# MAIN
# ============================================

def main():
    records = load_data()

    train_ids, test_ids = split_tracks(
        records
    )

    train = [
        r for r in records
        if r["track_id"] in train_ids
    ]

    test = [
        r for r in records
        if r["track_id"] in test_ids
    ]

    (
        global_scale,
        age_scales,
        counts
    ) = build_calibration(train)

    calibrated_train = apply_calibration(
        train, global_scale, age_scales
    )

    calibrated_test = apply_calibration(
        test, global_scale, age_scales
    )

    summaries = (
        evaluate_split(
            calibrated_train, "train"
        )
        + evaluate_split(
            calibrated_test, "test"
        )
    )

    scale_rows = [{
        "group": "global",
        "training_samples": sum(
            r["available"] for r in train
        ),
        "covariance_multiplier": global_scale
    }]

    for label, _, _ in AGE_BINS:
        scale_rows.append({
            "group": label,
            "training_samples": counts[label],
            "covariance_multiplier": (
                age_scales[label]
            )
        })

    save_csv(
        OUTPUT / "calibration_factors.csv",
        scale_rows
    )

    save_csv(
        OUTPUT / "calibration_summary.csv",
        summaries
    )

    save_csv(
        OUTPUT / "heldout_predictions.csv",
        calibrated_test
    )

    plot_coverage(summaries)
    plot_age(summaries)

    print(
        "\n========== PHASE 10: CALIBRATION =========="
    )

    print(
        f"Training track IDs: {len(train_ids)}"
    )

    print(
        f"Held-out track IDs: {len(test_ids)}"
    )

    print(
        f"Global covariance multiplier: "
        f"{global_scale:.3f}"
    )

    print("\nAge-dependent multipliers:")

    for label, _, _ in AGE_BINS:
        print(
            f"  {label:12} | "
            f"Samples={counts[label]:4} | "
            f"Scale={age_scales[label]:8.3f}"
        )

    print("\nHELD-OUT RESULTS")
    print("-------------------------------------------")

    for s in summaries:
        if s["split"] != "test":
            continue

        print(
            f"{s['group']:12} | "
            f"Available={s['coverage_pct']:6.2f}% | "
            f"Original={s['original_inside95_pct']:6.2f}% | "
            f"Global={s['global_inside95_pct']:6.2f}% | "
            f"Age={s['age_inside95_pct']:6.2f}%"
        )

    print("-------------------------------------------")
    print(f"Saved results: {OUTPUT}")
    print("===========================================")


if __name__ == "__main__":
    main()
