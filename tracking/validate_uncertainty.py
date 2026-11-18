
from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# VECTOR DYNAMICS - PHASE 8
# KALMAN UNCERTAINTY VALIDATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "tracking" / "persistent_results"
OUTPUT_DIR = INPUT_DIR / "uncertainty_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "baseline": INPUT_DIR / "baseline_tracks.csv",
    "network_test": INPUT_DIR / "network_test_tracks.csv"
}

PHASES = ["normal", "congested", "recovered"]

# 95% chi-square threshold with 2 degrees of freedom
CHI2_95_2D = 5.991464547

# Existing uncertainty column is:
# sqrt(Pxx + Pyy)
#
# For an isotropic covariance approximation:
# Pxx = Pyy = uncertainty^2 / 2
#
# Then the approximate 95% radial boundary is:
# sqrt(CHI2_95_2D / 2) * uncertainty

RADIAL_95_FACTOR = math.sqrt(CHI2_95_2D / 2.0)


# ==========================================
# HELPERS
# ==========================================

def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def percentile_or_nan(values, percentile):
    return (
        float(np.percentile(values, percentile))
        if values else float("nan")
    )


def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def age_category(age_ms):
    if age_ms < 100:
        return "0-100 ms"
    if age_ms < 200:
        return "100-200 ms"
    if age_ms < 300:
        return "200-300 ms"
    if age_ms < 400:
        return "300-400 ms"
    return "400+ ms"


def parse_file(path):
    records = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            valid = (
                row["error_px"] != ""
                and row["uncertainty_px"] != ""
                and row["age_ms"] != ""
            )

            record = {
                "scenario": row["scenario"],
                "phase": row["phase"],
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "status": row["status"],
                "valid": valid
            }

            if valid:
                error = float(row["error_px"])
                uncertainty = float(row["uncertainty_px"])
                age = float(row["age_ms"])

                radius95 = RADIAL_95_FACTOR * uncertainty

                record.update({
                    "error_px": error,
                    "uncertainty_px": uncertainty,
                    "age_ms": age,
                    "age_group": age_category(age),
                    "radius95_px": radius95,
                    "inside95": error <= radius95,
                    "normalized_error": (
                        error / uncertainty
                        if uncertainty > 0 else float("nan")
                    )
                })

            records.append(record)

    return records


# ==========================================
# LOAD DATA
# ==========================================

all_records = {}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 7 output: {path}"
        )

    all_records[name] = parse_file(path)


# ==========================================
# SUMMARIZE A GROUP
# ==========================================

def summarize(records, scenario, group_name):

    valid = [
        r for r in records if r["valid"]
    ]

    errors = [r["error_px"] for r in valid]
    uncertainty = [r["uncertainty_px"] for r in valid]
    ages = [r["age_ms"] for r in valid]

    inside = sum(r["inside95"] for r in valid)

    total = len(records)
    available = len(valid)

    return {
        "scenario": scenario,
        "group": group_name,
        "total_positions": total,
        "available_positions": available,
        "coverage_pct": round(
            100 * available / total, 2
        ) if total else "",
        "mean_error_px": round(
            mean_or_nan(errors), 3
        ),
        "median_error_px": round(
            percentile_or_nan(errors, 50), 3
        ),
        "p95_error_px": round(
            percentile_or_nan(errors, 95), 3
        ),
        "mean_uncertainty_px": round(
            mean_or_nan(uncertainty), 3
        ),
        "mean_age_ms": round(
            mean_or_nan(ages), 2
        ),
        "inside_approx_95_pct": round(
            100 * inside / available, 2
        ) if available else "",
        "unavailable_count": sum(
            r["status"] == "UNAVAILABLE"
            for r in records
        ),
        "expired_count": sum(
            r["status"] == "EXPIRED"
            for r in records
        )
    }


# ==========================================
# CREATE SUMMARY TABLES
# ==========================================

summary_rows = []
age_rows = []

for scenario, records in all_records.items():

    summary_rows.append(
        summarize(records, scenario, "overall")
    )

    for phase in PHASES:
        group = [
            r for r in records
            if r["phase"] == phase
        ]

        summary_rows.append(
            summarize(group, scenario, phase)
        )

    for age_group in [
        "0-100 ms",
        "100-200 ms",
        "200-300 ms",
        "300-400 ms",
        "400+ ms"
    ]:

        group = [
            r for r in records
            if r["valid"]
            and r["age_group"] == age_group
        ]

        if group:
            age_rows.append(
                summarize(
                    group, scenario, age_group
                )
            )


save_csv(
    OUTPUT_DIR / "uncertainty_summary.csv",
    summary_rows,
    list(summary_rows[0].keys())
)

if age_rows:
    save_csv(
        OUTPUT_DIR / "uncertainty_by_age.csv",
        age_rows,
        list(age_rows[0].keys())
    )


# ==========================================
# FIND WORST TRACK IDS
# ==========================================

worst_rows = []

for scenario, records in all_records.items():

    by_track = defaultdict(list)

    for r in records:
        if r["valid"]:
            by_track[r["track_id"]].append(r)

    for track_id, group in by_track.items():

        errors = [
            r["error_px"] for r in group
        ]

        worst_rows.append({
            "scenario": scenario,
            "track_id": track_id,
            "observations": len(group),
            "mean_error_px": round(
                float(np.mean(errors)), 3
            ),
            "max_error_px": round(
                float(np.max(errors)), 3
            ),
            "inside_approx_95_pct": round(
                100 * np.mean([
                    r["inside95"] for r in group
                ]), 2
            )
        })

worst_rows.sort(
    key=lambda r: r["mean_error_px"],
    reverse=True
)

save_csv(
    OUTPUT_DIR / "worst_tracks.csv",
    worst_rows,
    list(worst_rows[0].keys())
)


# ==========================================
# GRAPH 1: ERROR VS UNCERTAINTY
# ==========================================

network_valid = [
    r for r in all_records["network_test"]
    if r["valid"]
]

errors = np.array([
    r["error_px"] for r in network_valid
])

uncertainties = np.array([
    r["uncertainty_px"] for r in network_valid
])

plt.figure(figsize=(10, 6))

plt.scatter(
    uncertainties,
    errors,
    s=10,
    alpha=0.35
)

maximum = max(
    float(np.max(errors)),
    float(np.max(uncertainties))
)

x = np.linspace(0, maximum, 200)

plt.plot(
    x,
    RADIAL_95_FACTOR * x,
    linestyle="--",
    label="Approximate 95% radial boundary"
)

plt.xlabel("Reported uncertainty (pixels)")
plt.ylabel("Actual position error (pixels)")
plt.title("Vector Dynamics: Error vs Kalman Uncertainty")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_vs_uncertainty.png",
    dpi=200
)
plt.close()


# ==========================================
# GRAPH 2: ERROR BY MEASUREMENT AGE
# ==========================================

age_labels = [
    "0-100 ms",
    "100-200 ms",
    "200-300 ms",
    "300-400 ms",
    "400+ ms"
]

age_data = [
    [
        r["error_px"]
        for r in network_valid
        if r["age_group"] == label
    ]
    for label in age_labels
]

nonempty = [
    (label, values)
    for label, values in zip(age_labels, age_data)
    if values
]

plt.figure(figsize=(10, 6))

plt.boxplot(
    [values for _, values in nonempty],
    tick_labels=[
        label for label, _ in nonempty
    ],
    showfliers=False
)

plt.xlabel("Measurement age")
plt.ylabel("Position error (pixels)")
plt.title("Prediction Error vs Measurement Age")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_by_age.png",
    dpi=200
)
plt.close()


# ==========================================
# GRAPH 3: CALIBRATION BY NETWORK PHASE
# ==========================================

phase_coverage = []

for phase in PHASES:

    group = [
        r for r in network_valid
        if r["phase"] == phase
    ]

    phase_coverage.append(
        100 * np.mean([
            r["inside95"] for r in group
        ]) if group else float("nan")
    )

plt.figure(figsize=(9, 5))

plt.bar(PHASES, phase_coverage)

plt.axhline(
    95,
    linestyle="--",
    label="Nominal 95%"
)

plt.ylabel("Positions inside approximate boundary (%)")
plt.title("Approximate Uncertainty Calibration")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_by_phase.png",
    dpi=200
)
plt.close()


# ==========================================
# PRINT REPORT
# ==========================================

print("\n========== UNCERTAINTY VALIDATION ==========")

for row in summary_rows:

    print(
        f"{row['scenario']:12} | "
        f"{row['group']:10} | "
        f"Coverage={row['coverage_pct']:6.2f}% | "
        f"Mean={row['mean_error_px']:7.2f} px | "
        f"Inside95={str(row['inside_approx_95_pct']):>6}%"
    )

print("--------------------------------------------")

print("Worst five tracks by mean error:")

for row in worst_rows[:5]:

    print(
        f"Track {row['track_id']:4} | "
        f"{row['scenario']:12} | "
        f"Mean={row['mean_error_px']:7.2f} px | "
        f"Max={row['max_error_px']:7.2f} px"
    )

print("--------------------------------------------")
print(f"Saved: {OUTPUT_DIR}")
print("============================================")
