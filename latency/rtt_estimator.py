
from pathlib import Path
import csv
import random
import statistics
import math

# ==========================================
# VECTOR DYNAMICS
# JACOBSON-KARELS RTT ESTIMATOR
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "latency" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "rtt_results.csv"

# Jacobson-Karels parameters
ALPHA = 1.0 / 8.0
BETA = 1.0 / 4.0
K = 4.0

# Minimum retransmission timeout (milliseconds)
MIN_RTO_MS = 100.0

# Simulated communication settings
SAMPLES = 300
BASE_ONE_WAY_DELAY_MS = 40.0
JITTER_STD_MS = 8.0

# Repeatable simulation
random.seed(42)


# ==========================================
# RTT ESTIMATOR
# ==========================================

class RTTEstimator:

    def __init__(self):
        self.srtt = None
        self.rttvar = None
        self.rto = None

    def update(self, measured_rtt_ms):

        if measured_rtt_ms <= 0:
            raise ValueError("RTT must be positive")

        if self.srtt is None:

            # Standard first-sample initialization
            self.srtt = measured_rtt_ms
            self.rttvar = measured_rtt_ms / 2.0

        else:

            # RFC 6298 update order:
            # calculate variation using OLD SRTT
            old_srtt = self.srtt

            self.rttvar = (
                (1.0 - BETA) * self.rttvar
                + BETA * abs(old_srtt - measured_rtt_ms)
            )

            self.srtt = (
                (1.0 - ALPHA) * old_srtt
                + ALPHA * measured_rtt_ms
            )

        self.rto = max(
            MIN_RTO_MS,
            self.srtt + K * self.rttvar
        )

        return {
            "srtt_ms": self.srtt,
            "rttvar_ms": self.rttvar,
            "rto_ms": self.rto
        }


# ==========================================
# SIMULATE NETWORK CONDITIONS
# ==========================================

def simulate_one_way_delay(sample_index):

    # Normal communication
    if sample_index < 100:
        base_delay = BASE_ONE_WAY_DELAY_MS
        jitter = JITTER_STD_MS
        condition = "normal"

    # Increased latency and jitter
    elif sample_index < 200:
        base_delay = 100.0
        jitter = 25.0
        condition = "congested"

    # Recovery
    else:
        base_delay = BASE_ONE_WAY_DELAY_MS
        jitter = JITTER_STD_MS
        condition = "recovered"

    delay = random.gauss(base_delay, jitter)

    # One-way delay cannot be negative
    return max(1.0, delay), condition


# ==========================================
# RUN SIMULATION
# ==========================================

estimator = RTTEstimator()
results = []

for i in range(SAMPLES):

    # Simulate outbound and return trips
    outbound_ms, condition = simulate_one_way_delay(i)
    return_ms, _ = simulate_one_way_delay(i)

    # Simplified RTT model:
    # outbound + return, no remote processing delay
    measured_rtt = outbound_ms + return_ms

    estimates = estimator.update(measured_rtt)

    results.append({
        "sample": i + 1,
        "condition": condition,
        "outbound_ms": round(outbound_ms, 3),
        "return_ms": round(return_ms, 3),
        "measured_rtt_ms": round(measured_rtt, 3),
        "srtt_ms": round(estimates["srtt_ms"], 3),
        "rttvar_ms": round(estimates["rttvar_ms"], 3),
        "rto_ms": round(estimates["rto_ms"], 3)
    })


# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# PRINT REPORT
# ==========================================

print("\n========== RTT ESTIMATOR REPORT ==========")

for condition in ["normal", "congested", "recovered"]:

    group = [
        row for row in results
        if row["condition"] == condition
    ]

    mean_rtt = statistics.mean(
        row["measured_rtt_ms"] for row in group
    )

    mean_srtt = statistics.mean(
        row["srtt_ms"] for row in group
    )

    mean_rto = statistics.mean(
        row["rto_ms"] for row in group
    )

    print(
        f"{condition.capitalize():10} | "
        f"RTT={mean_rtt:7.2f} ms | "
        f"SRTT={mean_srtt:7.2f} ms | "
        f"RTO={mean_rto:7.2f} ms"
    )

print("------------------------------------------")
print(f"Samples processed: {len(results)}")
print(f"Final SRTT: {estimator.srtt:.2f} ms")
print(f"Final RTTVAR: {estimator.rttvar:.2f} ms")
print(f"Final RTO: {estimator.rto:.2f} ms")
print(f"Saved: {OUTPUT_CSV}")
print("==========================================")
