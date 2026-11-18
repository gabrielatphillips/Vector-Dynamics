"""Vector Dynamics: existing fused target snapshots -> 32-byte packets -> cab replay.
Run from project root: python communication/fused_cab_pipeline.py --display
Uses fusion/target_manager.py + fusion/radar_association.py (existing algorithms).
All radar values are SYNTHETIC. No safety warnings or safe-passing decisions.
"""
from __future__ import annotations
import argparse, csv, heapq, random, subprocess, sys
from collections import Counter
from pathlib import Path
from vehicle_protocol import (Module, Status, VehiclePacket, Receiver, encode,
    RANGE_VALID, VELOCITY_VALID, BEARING_VALID, CAMERA_FRESH, RADAR_FRESH, SYNTHETIC)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "fused_results"
SNAPSHOTS = ROOT / "fusion" / "phase13_results" / "target_snapshots.csv"
FIELDS = ["time_ms","front_fresh","front_track_id","front_age_ms","front_range_ft",
          "front_closing_mph","front_source","rear_fresh","rear_track_id","rear_age_ms",
          "rear_range_ft","rear_closing_mph","rear_source"]

def generate_snapshots():
    manager = ROOT / "fusion" / "target_manager.py"
    assoc = ROOT / "fusion" / "radar_association.py"
    if not manager.is_file() or not assoc.is_file():
        raise FileNotFoundError("Expected fusion/target_manager.py and fusion/radar_association.py. "
                                "Copy your existing fusion modules there (without phase numbers).")
    print("Running existing radar-camera association and target manager...")
    subprocess.run([sys.executable, str(manager)], cwd=str(ROOT), check=True)
    if not SNAPSHOTS.is_file():
        raise FileNotFoundError(f"Target manager did not generate {SNAPSHOTS}")

def load_snapshots(path, duration):
    frames = {}
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"time_s","fused_id","status","source","radar_valid",
                    "camera_valid","range_m","radial_velocity_mps","angle_deg"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing target snapshot columns: {required-set(reader.fieldnames or [])}")
        for row in reader:
            t = float(row["time_s"])
            if t < 0 or t >= duration or row["status"] != "confirmed":
                continue
            tick = int(t * 10 + 1e-6)
            frames.setdefault(tick, []).append(row)
            count += 1
    if not frames:
        raise ValueError("No confirmed fused snapshots in selected duration")
    return frames, count

def packet_from_snapshot(row, seq, send_ms):
    radar = row["radar_valid"].lower() in ("true","1")
    camera = row["camera_valid"].lower() in ("true","1")
    flags = SYNTHETIC | (CAMERA_FRESH if camera else 0) | (RADAR_FRESH if radar else 0)
    has_metric = radar and row["range_m"] and row["radial_velocity_mps"] and row["angle_deg"]
    if has_metric:
        flags |= RANGE_VALID | VELOCITY_VALID | BEARING_VALID
    return VehiclePacket(Module.FRONT, Module.CAB, seq, send_ms,
        int(row["fused_id"]), round(float(row["range_m"])*1000) if has_metric else 0,
        round(float(row["radial_velocity_mps"])*1000) if has_metric else 0,
        round(float(row["angle_deg"])*100) if has_metric else 0, flags, Status.CONFIRMED)

def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def run(snapshot_path=SNAPSHOTS, duration=20.0, seed=42, drop_rate=.10,
        corrupt_rate=.04, freshness_ms=300):
    if duration <= 0 or freshness_ms <= 0 or not 0 <= drop_rate <= 1 or not 0 <= corrupt_rate <= 1:
        raise ValueError("Invalid duration, freshness, or error rates")
    frames, n = load_snapshots(Path(snapshot_path), duration)
    rng = random.Random(seed)
    receiver = Receiver(max_age_ms=freshness_ms)
    seq = {Module.FRONT:0, Module.REAR:0}
    queue, logs, timeline, serial = [], [], [], 0
    stats = Counter()
    duration_ms = int(duration*1000)
    # One target per module per 100-ms slot: packet protocol is single-target.
    for ms in range(0, duration_ms, 100):
        tick = ms//100
        packets = []
        choices = frames.get(tick, [])
        # Prefer confirmed camera+radar tracks, then radar-only, then camera-only.
        if choices:
            def priority(r):
                metric = r["radar_valid"].lower() in ("true","1") and bool(r["range_m"])
                fused = metric and r["camera_valid"].lower() in ("true","1")
                return (2 if fused else 1 if metric else 0,
                        -float(r["range_m"]) if metric else -9999,
                        -int(r["fused_id"]))
            best = max(choices, key=priority)
            packets.append(packet_from_snapshot(best, seq[Module.FRONT], ms))
            stats["front_selected"] += 1
            if best["source"] == "fused": stats["fused_selected"] += 1
        # Rear source remains synthetic until actual rear radar integration.
        if not 9000 <= ms < 13000:
            packets.append(VehiclePacket(Module.REAR, Module.CAB, seq[Module.REAR],
                ms, 22, max(5000, 42000-ms//1000), -1000, 0,
                RANGE_VALID|VELOCITY_VALID|RADAR_FRESH|SYNTHETIC, Status.CONFIRMED))
        else:
            stats["rear_outage_ticks"] += 1
        for module in seq:
            seq[module] += 1
        for pkt in packets:
            stats["sent"] += 1
            if rng.random() < drop_rate:
                stats["dropped"] += 1
                continue
            raw = bytearray(encode(pkt))
            if rng.random() < corrupt_rate:
                raw[rng.randrange(len(raw)-4)] ^= 1
                stats["corrupted"] += 1
            arrival = ms + rng.randint(20, 120)
            heapq.heappush(queue, (arrival, serial, bytes(raw)))
            serial += 1
    for now in range(0, duration_ms+freshness_ms+200, 50):
        while queue and queue[0][0] <= now:
            arrival, _, raw = heapq.heappop(queue)
            try:
                pkt = receiver.ingest(raw, arrival)
                result = "accepted"
                stats["accepted"] += 1
                source = pkt.source.name.lower()
            except ValueError:
                result = "rejected"
                stats["rejected"] += 1
                source = "unknown"
            logs.append({"arrival_ms":arrival,"source":source,"result":result,"packet_hex":raw.hex()})
        active = receiver.current(now)
        row = {"time_ms":now}
        for module, name in ((Module.FRONT,"front"),(Module.REAR,"rear")):
            candidates = [(key,pkt) for key,pkt in active.items() if key[0] == module]
            if candidates:
                key,pkt = max(candidates, key=lambda pair: receiver.last_arrival[pair[0]])
                age = now-receiver.last_arrival[key]
                valid_range = bool(pkt.flags & RANGE_VALID)
                valid_speed = bool(pkt.flags & VELOCITY_VALID)
                source = ("FUSED (SYNTHETIC RADAR)" if pkt.flags & CAMERA_FRESH and pkt.flags & RADAR_FRESH
                          else "SYNTHETIC RADAR" if pkt.flags & RADAR_FRESH
                          else "VIDEO TRACK ONLY")
                row.update({f"{name}_fresh":1,f"{name}_track_id":pkt.track_id,
                    f"{name}_age_ms":age,
                    f"{name}_range_ft":f"{pkt.range_mm/1000*3.28084:.1f}" if valid_range else "",
                    f"{name}_closing_mph":f"{max(0,-pkt.radial_mmps/1000*2.23694):.1f}" if valid_speed else "",
                    f"{name}_source":source})
            else:
                row.update({f"{name}_fresh":0,f"{name}_track_id":"",
                    f"{name}_age_ms":"","%s_range_ft"%name:"",
                    f"{name}_closing_mph":"","%s_source"%name:"NO FRESH DATA"})
        timeline.append(row)
    assert stats["sent"] == stats["accepted"]+stats["rejected"]+stats["dropped"]
    assert all(not row["rear_fresh"] for row in timeline if 9300 <= row["time_ms"] < 13000)
    assert all(not row["front_range_ft"] for row in timeline if not row["front_fresh"])
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"packet_log.csv",logs,["arrival_ms","source","result","packet_hex"])
    write_csv(OUT/"cab_timeline.csv",timeline,FIELDS)
    print("========== FUSED VIDEO-TO-CAB SIMULATION ==========")
    print(f"Confirmed snapshot rows: {n} | Front slots selected: {stats['front_selected']} | Fused selected: {stats['fused_selected']}")
    print(f"Sent={stats['sent']} Accepted={stats['accepted']} Dropped={stats['dropped']} Rejected={stats['rejected']}")
    print("Rear outage and stale-data suppression: PASS")
    print(f"Results: {OUT}")
    print("Synthetic radar distances/speeds; NOT calibrated measurements. NO safety warnings authorized.")
    return timeline

def display(speed=1.0):
    import tkinter as tk
    from tkinter import ttk
    if speed <= 0: raise ValueError("Speed must be positive")
    path = OUT/"cab_timeline.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run without --display first")
    with path.open(newline="",encoding="utf-8") as f:
        timeline = list(csv.DictReader(f))
    root=tk.Tk()
    root.title("Vector Dynamics | Fused Cab Simulation")
    root.geometry("900x430")
    root.configure(bg="#111827")
    fg, muted, panel="#f9fafb","#9ca3af","#1f2937"
    tk.Label(root,text="VECTOR DYNAMICS  |  FUSED SENSOR REPLAY",bg="#111827",fg=fg,font=("Segoe UI",17,"bold")).pack(pady=(20,6))
    tk.Label(root,text="RADAR SYNTHETIC  •  VIDEO TRACKS REAL  •  NO SAFETY WARNINGS",bg="#111827",fg="#fbbf24").pack()
    clock=tk.StringVar()
    tk.Label(root,textvariable=clock,bg="#111827",fg=muted).pack(pady=8)
    cards=tk.Frame(root,bg="#111827")
    cards.pack(fill="both",expand=True,padx=20,pady=10)
    variables={}
    for col,name in enumerate(("front","rear")):
        frame=tk.Frame(cards,bg=panel,padx=18,pady=14)
        frame.grid(row=0,column=col,padx=7,sticky="nsew")
        tk.Label(frame,text=name.upper(),bg=panel,fg=fg,font=("Segoe UI",16,"bold")).pack(anchor="w")
        vars_=[tk.StringVar() for _ in range(4)]
        for i,var in enumerate(vars_):
            tk.Label(frame,textvariable=var,bg=panel,fg=fg if i<2 else muted,
                     font=("Segoe UI",15 if i==0 else 11)).pack(anchor="w",pady=4)
        variables[name]=vars_
    for col in (0,1): cards.grid_columnconfigure(col,weight=1,uniform="card")
    cards.grid_rowconfigure(0,weight=1)
    progress=ttk.Progressbar(root,maximum=len(timeline))
    progress.pack(fill="x",padx=26,pady=(0,12))
    index=[0]
    def tick():
        if index[0]>=len(timeline): return
        row=timeline[index[0]]
        clock.set(f"Replay time: {int(row['time_ms'])/1000:.2f} s")
        for name,vars_ in variables.items():
            fresh=int(row[f"{name}_fresh"])
            vars_[0].set(f"Track ID: {row[f'{name}_track_id']}" if fresh else "NO FRESH DATA")
            vars_[1].set(f"Distance: {row[f'{name}_range_ft']} ft" if row[f"{name}_range_ft"] else "Distance: unavailable")
            vars_[2].set(f"Closing: {row[f'{name}_closing_mph']} mph" if row[f"{name}_closing_mph"] else "Closing: unavailable")
            vars_[3].set(f"{row[f'{name}_source']}  |  Age: {row[f'{name}_age_ms'] or '—'} ms")
        index[0]+=1
        progress["value"]=index[0]
        root.after(max(1,round(50/speed)),tick)
    tick()
    root.mainloop()

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots",type=Path,default=SNAPSHOTS)
    ap.add_argument("--generate",action="store_true",help="Run existing fusion target manager before integration")
    ap.add_argument("--display",action="store_true")
    ap.add_argument("--speed",type=float,default=1.0)
    ap.add_argument("--duration",type=float,default=20.0)
    args=ap.parse_args()
    if args.generate: generate_snapshots()
    if args.display:
        if not (OUT/"cab_timeline.csv").exists():
            run(args.snapshots,args.duration)
        display(args.speed)
    else:
        run(args.snapshots,args.duration)
