"""Vector Dynamics virtual cab display: replay accepted network packets.

Run: python communication/cab_display.py
Headless validation: python communication/cab_display.py --check
Requires network_simulator.py and vehicle_protocol.py in the same directory.
Synthetic demonstration ONLY. No safety warnings are authorized.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from network_simulator import packet_for
from vehicle_protocol import Module, Receiver, encode, SYNTHETIC, RANGE_VALID, VELOCITY_VALID

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "network_results"
FRESHNESS_MS = 300
TICK_MS = 50


def read_events():
    path = RESULTS / "packet_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python communication/network_simulator.py first.")
    events = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["result"] != "accepted":
                continue
            module = Module.FRONT if row["source"] == "front" else Module.REAR
            events.append((int(row["arrival_ms"]), module, int(row["send_ms"])))
    return sorted(events, key=lambda x: x[0])


def read_timeline():
    path = RESULTS / "cab_timeline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python communication/network_simulator.py first.")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Replay:
    def __init__(self, events):
        self.events = events
        self.index = 0
        self.receiver = Receiver(max_age_ms=FRESHNESS_MS)
        self.last_send_ms = {}

    def advance(self, now_ms):
        while self.index < len(self.events) and self.events[self.index][0] <= now_ms:
            arrival, module, send_ms = self.events[self.index]
            self.index += 1
            # Reconstruct exactly the deterministic synthetic payload generated
            # by network_simulator.py. Packet log contains no payload bytes.
            pkt = packet_for(module, send_ms // 100, send_ms)
            self.receiver.ingest(encode(pkt), arrival)
            self.last_send_ms[module] = send_ms
        active = self.receiver.current(now_ms)
        states = {}
        for module in (Module.FRONT, Module.REAR):
            pkt = active.get((module, 11 if module == Module.FRONT else 22))
            last_arrival = self.receiver.last_arrival.get((module, 11 if module == Module.FRONT else 22))
            age = None if last_arrival is None else now_ms - last_arrival
            usable = (pkt is not None and pkt.flags & SYNTHETIC
                      and pkt.flags & RANGE_VALID and pkt.flags & VELOCITY_VALID)
            states[module] = {
                "fresh": bool(usable),
                "age_ms": age,
                "range_ft": (pkt.range_mm / 1000 * 3.28084) if usable else None,
                "closing_mph": max(0.0, -pkt.radial_mmps / 1000 * 2.23694) if usable else None,
                "track_id": pkt.track_id if usable else None,
            }
        return states


def validate():
    events, timeline = read_events(), read_timeline()
    replay = Replay(events)
    checks = 0
    outages = 0
    for row in timeline:
        now = int(row["time_ms"])
        state = replay.advance(now)
        for module, name in ((Module.FRONT, "front"), (Module.REAR, "rear")):
            expected = bool(int(row[f"{name}_fresh"]))
            assert state[module]["fresh"] == expected, (now, name, expected, state[module])
            checks += 1
        if 9300 <= now < 13000:
            assert not state[Module.REAR]["fresh"], f"Rear unexpectedly fresh at {now} ms"
            outages += 1
    print("========== CAB DISPLAY REPLAY CHECK ==========")
    print(f"Accepted packet events: {len(events)}")
    print(f"Timeline rows: {len(timeline)} | Freshness comparisons: {checks}")
    print(f"Rear outage stale checks: {outages}")
    print("Replay agrees with saved cab timeline: PASS")
    print("Synthetic telemetry only; safety warnings NOT authorized")


def launch(speed=1.0):
    import tkinter as tk
    from tkinter import ttk

    events = read_events()
    timeline = read_timeline()
    if not timeline:
        raise ValueError("cab_timeline.csv has no rows")
    replay = Replay(events)
    root = tk.Tk()
    root.title("Vector Dynamics | Virtual Cab")
    root.geometry("760x460")
    root.configure(bg="#111827")
    root.minsize(640, 410)
    fg, muted, bg, panel = "#f9fafb", "#9ca3af", "#111827", "#1f2937"

    header = tk.Frame(root, bg=bg)
    header.pack(fill="x", padx=24, pady=(20, 8))
    tk.Label(header, text="VECTOR DYNAMICS", fg=fg, bg=bg,
             font=("Segoe UI", 20, "bold")).pack(side="left")
    tk.Label(header, text="SYNTHETIC REPLAY · NO SAFETY WARNINGS", fg="#fbbf24",
             bg=bg, font=("Segoe UI", 10, "bold")).pack(side="right")

    clock = tk.StringVar(value="Time 0.00 s")
    tk.Label(root, textvariable=clock, fg=muted, bg=bg,
             font=("Segoe UI", 12)).pack(anchor="w", padx=26, pady=(0, 12))

    body = tk.Frame(root, bg=bg)
    body.pack(fill="both", expand=True, padx=20)
    widgets = {}
    for i, (module, label) in enumerate(((Module.FRONT, "FRONT"), (Module.REAR, "REAR"))):
        card = tk.Frame(body, bg=panel, padx=20, pady=20)
        card.grid(row=0, column=i, sticky="nsew", padx=6)
        tk.Label(card, text=label, fg=fg, bg=panel,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        distance = tk.StringVar(value="— ft")
        tk.Label(card, textvariable=distance, fg=fg, bg=panel,
                 font=("Segoe UI", 32, "bold")).pack(anchor="w", pady=(12, 4))
        closing = tk.StringVar(value="Closing: —")
        tk.Label(card, textvariable=closing, fg=muted, bg=panel,
                 font=("Segoe UI", 13)).pack(anchor="w")
        status = tk.StringVar(value="UNAVAILABLE")
        status_label = tk.Label(card, textvariable=status, fg="#f87171", bg=panel,
                                font=("Segoe UI", 12, "bold"))
        status_label.pack(anchor="w", pady=(20, 2))
        age = tk.StringVar(value="Last packet: —")
        tk.Label(card, textvariable=age, fg=muted, bg=panel,
                 font=("Segoe UI", 11)).pack(anchor="w")
        widgets[module] = (distance, closing, status, status_label, age)
    body.grid_columnconfigure(0, weight=1, uniform="modules")
    body.grid_columnconfigure(1, weight=1, uniform="modules")
    body.grid_rowconfigure(0, weight=1)

    controls = tk.Frame(root, bg=bg)
    controls.pack(fill="x", padx=26, pady=18)
    progress = ttk.Progressbar(controls, maximum=max(1, len(timeline) - 1))
    progress.pack(fill="x", pady=(0, 10))
    paused = tk.BooleanVar(value=False)
    pause_btn = tk.Button(controls, text="Pause", width=12)
    pause_btn.pack(side="left")
    tk.Label(controls, text="Replay speed: %.1fx" % speed,
             fg=muted, bg=bg).pack(side="right")

    index = [0]

    def toggle_pause():
        paused.set(not paused.get())
        pause_btn.configure(text="Resume" if paused.get() else "Pause")
    pause_btn.configure(command=toggle_pause)

    def tick():
        if not paused.get() and index[0] < len(timeline):
            row = timeline[index[0]]
            now = int(row["time_ms"])
            states = replay.advance(now)
            clock.set(f"Simulation time: {now / 1000:.2f} s")
            progress["value"] = index[0]
            for module in (Module.FRONT, Module.REAR):
                state = states[module]
                distance, closing, status, status_label, age = widgets[module]
                if state["fresh"]:
                    distance.set(f"{state['range_ft']:.0f} ft")
                    closing.set(f"Closing: {state['closing_mph']:.1f} mph")
                    status.set("TELEMETRY FRESH")
                    status_label.configure(fg="#4ade80")
                else:
                    distance.set("— ft")
                    closing.set("Closing: —")
                    status.set("NO FRESH DATA")
                    status_label.configure(fg="#f87171")
                age.set("Last packet: —" if state["age_ms"] is None
                        else f"Last packet: {state['age_ms']} ms ago")
            index[0] += 1
        if index[0] < len(timeline):
            root.after(max(1, round(TICK_MS / speed)), tick)
        else:
            pause_btn.configure(state="disabled", text="Replay ended")
    tick()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate replay against saved cab timeline without GUI")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.check:
        validate()
    else:
        launch(args.speed)
