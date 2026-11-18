"""Vector Dynamics: deterministic three-node virtual ESP32 telemetry network.

Run from project root: python communication/network_simulator.py
Requires communication/vehicle_protocol.py. No third-party dependencies.
All values are synthetic. No safety decisions are authorized.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import random
from collections import Counter
from pathlib import Path

from vehicle_protocol import (
    Module, Status, VehiclePacket, Receiver, encode, decode,
    RANGE_VALID, VELOCITY_VALID, BEARING_VALID, RADAR_FRESH, SYNTHETIC,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "communication" / "network_results"
FLAGS = RANGE_VALID | VELOCITY_VALID | BEARING_VALID | RADAR_FRESH | SYNTHETIC


def packet_for(module, seq, time_ms):
    if module == Module.FRONT:
        # Synthetic oncoming target: positive range, negative radial velocity.
        distance = max(5000, 110000 - (time_ms * 3) // 1000)
        speed = -3000
        bearing = -350
        track_id = 11
    else:
        distance = max(5000, 42000 - (time_ms * 1) // 1000)
        speed = -1000
        bearing = 250
        track_id = 22
    return VehiclePacket(module, Module.CAB, seq, time_ms, track_id,
                         distance, speed, bearing, FLAGS, Status.CONFIRMED)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def simulate(duration_s=20, period_ms=100, seed=42, drop_rate=0.10,
             corrupt_rate=0.04, delay_min_ms=20, delay_max_ms=120,
             freshness_ms=300, disconnect_start_s=9, disconnect_end_s=13):
    if duration_s <= 0 or period_ms <= 0 or freshness_ms <= 0:
        raise ValueError("Duration, period and freshness must be positive")
    if not (0 <= drop_rate <= 1 and 0 <= corrupt_rate <= 1):
        raise ValueError("Rates must be between zero and one")
    if not (0 <= delay_min_ms <= delay_max_ms):
        raise ValueError("Invalid delay bounds")
    if not (0 <= disconnect_start_s < disconnect_end_s <= duration_s):
        raise ValueError("Disconnect interval must be inside simulation")

    rng = random.Random(seed)
    receiver = Receiver(max_age_ms=freshness_ms)
    counters = Counter()
    by_source = {module: Counter() for module in (Module.FRONT, Module.REAR)}
    events = []
    serial = 0
    duration_ms = int(duration_s * 1000)
    down_start = int(disconnect_start_s * 1000)
    down_end = int(disconnect_end_s * 1000)

    # Schedule sender-side traffic independently for each module.
    for send_ms in range(0, duration_ms, period_ms):
        for module in (Module.FRONT, Module.REAR):
            seq = send_ms // period_ms
            counters["sent"] += 1
            by_source[module]["sent"] += 1
            if module == Module.REAR and down_start <= send_ms < down_end:
                counters["disconnected"] += 1
                by_source[module]["disconnected"] += 1
                continue
            if rng.random() < drop_rate:
                counters["dropped"] += 1
                by_source[module]["dropped"] += 1
                continue
            raw = bytearray(encode(packet_for(module, seq, send_ms)))
            if rng.random() < corrupt_rate:
                pos = rng.randrange(len(raw) - 4)
                raw[pos] ^= 1 << rng.randrange(8)
                counters["corrupted"] += 1
                by_source[module]["corrupted"] += 1
            delay = rng.randint(delay_min_ms, delay_max_ms)
            arrival_ms = send_ms + delay
            heapq.heappush(events, (arrival_ms, serial, module, send_ms, bytes(raw)))
            serial += 1

    # Event-driven cab receiver: assess freshness at regular wall-clock ticks,
    # not only after all packets have been delivered.
    packet_log = []
    timeline = []
    tick_ms = 50
    for now_ms in range(0, duration_ms + freshness_ms + delay_max_ms + tick_ms, tick_ms):
        while events and events[0][0] <= now_ms:
            arrival, _, module, send_ms, raw = heapq.heappop(events)
            try:
                decoded = decode(raw)
                if decoded.destination != Module.CAB or decoded.source != module:
                    raise ValueError("Wrong destination or source")
                # Important: the existing Receiver only checks ordering; a
                # packet may be old on arrival. Reject via simulated delivery
                # delay (ground truth), not unsynchronized sender clock.
                if arrival - send_ms > freshness_ms:
                    raise ValueError("Exceeded simulated delivery-age limit")
                receiver.ingest(raw, arrival)
                counters["accepted"] += 1
                by_source[module]["accepted"] += 1
                result = "accepted"
            except ValueError as exc:
                result = "rejected: " + str(exc)
                counters["rejected"] += 1
                by_source[module]["rejected"] += 1
            packet_log.append({
                "source": module.name.lower(), "send_ms": send_ms,
                "arrival_ms": arrival, "latency_ms": arrival - send_ms,
                "result": result,
            })
        active = receiver.current(now_ms)
        front_fresh = any(src == Module.FRONT for src, _ in active)
        rear_fresh = any(src == Module.REAR for src, _ in active)
        timeline.append({"time_ms": now_ms, "front_fresh": int(front_fresh),
                         "rear_fresh": int(rear_fresh),
                         "front_state": "fresh" if front_fresh else "unavailable",
                         "rear_state": "fresh" if rear_fresh else "unavailable"})

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "packet_log.csv", packet_log,
              ["source", "send_ms", "arrival_ms", "latency_ms", "result"])
    write_csv(OUT / "cab_timeline.csv", timeline,
              ["time_ms", "front_fresh", "rear_fresh", "front_state", "rear_state"])
    summary = []
    for module in (Module.FRONT, Module.REAR):
        c = by_source[module]
        summary.append({"module": module.name.lower(), "sent": c["sent"],
                        "accepted": c["accepted"], "dropped": c["dropped"],
                        "corrupted": c["corrupted"], "disconnected": c["disconnected"],
                        "rejected": c["rejected"]})
    write_csv(OUT / "summary.csv", summary,
              ["module", "sent", "accepted", "dropped", "corrupted", "disconnected", "rejected"])

    outage_ticks = [r for r in timeline if down_start + freshness_ms <= r["time_ms"] < down_end]
    assert outage_ticks and all(r["rear_fresh"] == 0 for r in outage_ticks), \
        "Rear should become unavailable during extended disconnection"
    assert all(not (r["rear_fresh"] and r["rear_state"] == "unavailable") for r in timeline)
    assert counters["sent"] == (counters["accepted"] + counters["rejected"]
                                + counters["dropped"] + counters["disconnected"])
    print("========== VIRTUAL ESP32 NETWORK ==========")
    print(f"Packet size: 32 bytes | Duration: {duration_s}s | Tick: {tick_ms}ms")
    for row in summary:
        print(f"{row['module']:5} | Sent={row['sent']:3} | Accepted={row['accepted']:3} | "
              f"Dropped={row['dropped']:3} | Corrupted={row['corrupted']:3} | "
              f"Disconnected={row['disconnected']:3} | Rejected={row['rejected']:3}")
    print(f"Cab freshness during rear outage: PASS ({len(outage_ticks)} checked ticks)")
    print(f"Packet logs and cab timeline saved: {OUT}")
    print("Synthetic telemetry only; safety warnings NOT authorized")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simulate(duration_s=args.duration, seed=args.seed)
