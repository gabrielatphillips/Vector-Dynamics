"""Vector Dynamics: video-derived front tracks -> binary network -> virtual cab.

Uses precomputed YOLO/ByteTrack CSV, NOT live inference. Front video has no
calibrated metric range/velocity; those fields are intentionally invalid.
Rear telemetry is explicitly synthetic. No safety decisions are authorized.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import random
from collections import Counter
from pathlib import Path

from vehicle_protocol import (Module, Status, VehiclePacket, Receiver, encode,
                              RANGE_VALID, VELOCITY_VALID, RADAR_FRESH,
                              CAMERA_FRESH, SYNTHETIC)

HERE = Path(__file__).resolve().parent
DEFAULT_TRACKS = HERE.parent / 'vision' / 'results' / 'tracking_data.csv'
OUT = HERE / 'integrated_results'


def load_tracks(path, duration_s=20.0):
    if not path.is_file():
        raise FileNotFoundError(f'Tracking CSV not found: {path}\nPass --tracks with the path to your tracking_data.csv')
    by_tick = {}
    count = 0
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        needed = {'timestamp_s', 'track_id', 'confidence'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'Tracking CSV requires columns: {sorted(needed)}')
        for row in reader:
            timestamp = float(row['timestamp_s'])
            if not 0 <= timestamp < duration_s:
                continue
            tid = int(row['track_id'])
            if not 0 <= tid <= 65535:
                continue
            # Aggregate to 100 ms sender ticks; select the most confident track.
            tick = int(timestamp * 10 + 1e-7)
            confidence = float(row['confidence'])
            if tick not in by_tick or confidence > by_tick[tick][0]:
                by_tick[tick] = (confidence, tid)
            count += 1
    if not by_tick:
        raise ValueError('No usable tracks within selected duration')
    return by_tick, count


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(tracks_path=DEFAULT_TRACKS, duration_s=20.0, seed=42,
        drop_rate=0.10, corrupt_rate=0.04, freshness_ms=300):
    if duration_s <= 0 or freshness_ms <= 0:
        raise ValueError('Duration and freshness must be positive')
    if not (0 <= drop_rate <= 1 and 0 <= corrupt_rate <= 1):
        raise ValueError('Rates must be between 0 and 1')
    ticks, observations = load_tracks(Path(tracks_path), duration_s)
    rng = random.Random(seed)
    receiver = Receiver(max_age_ms=freshness_ms)
    counters = Counter()
    queue = []
    seq = {Module.FRONT: 0, Module.REAR: 0}
    serial = 0
    duration_ms = int(duration_s * 1000)
    for send_ms in range(0, duration_ms, 100):
        tick = send_ms // 100
        # Front: actual tracked ID, but NO distance/speed inference from pixels.
        packets = []
        if tick in ticks:
            _, tid = ticks[tick]
            packets.append(VehiclePacket(Module.FRONT, Module.CAB, seq[Module.FRONT],
                                         send_ms, tid, 0, 0, 0,
                                         CAMERA_FRESH | SYNTHETIC, Status.CONFIRMED))
        # Rear: entirely synthetic, to exercise two-source communications.
        if not (9000 <= send_ms < 13000):
            packets.append(VehiclePacket(Module.REAR, Module.CAB, seq[Module.REAR],
                                         send_ms, 22, max(5000, 42000 - send_ms // 1000),
                                         -1000, 0,
                                         RANGE_VALID | VELOCITY_VALID | RADAR_FRESH | SYNTHETIC,
                                         Status.CONFIRMED))
        else:
            counters['rear_outage_ticks'] += 1
        for module in (Module.FRONT, Module.REAR):
            seq[module] += 1  # Sequence continues across missing tracks/outages.
        for pkt in packets:
            counters['sent'] += 1
            if rng.random() < drop_rate:
                counters['dropped'] += 1
                continue
            raw = bytearray(encode(pkt))
            if rng.random() < corrupt_rate:
                raw[rng.randrange(len(raw) - 4)] ^= 1
                counters['corrupted'] += 1
            arrival = send_ms + rng.randint(20, 120)
            heapq.heappush(queue, (arrival, serial, bytes(raw)))
            serial += 1

    log, timeline = [], []
    for now in range(0, duration_ms + freshness_ms + 200, 50):
        while queue and queue[0][0] <= now:
            arrival, _, raw = heapq.heappop(queue)
            try:
                pkt = receiver.ingest(raw, arrival)
                result = 'accepted'
                counters['accepted'] += 1
                source = pkt.source.name.lower()
            except ValueError:
                result = 'rejected'
                counters['rejected'] += 1
                source = 'unknown'
            log.append(dict(arrival_ms=arrival, source=source,
                            result=result, packet_hex=raw.hex()))
        active = receiver.current(now)
        states = {}
        for module in (Module.FRONT, Module.REAR):
            candidates = [(key, pkt) for key, pkt in active.items() if key[0] == module]
            if candidates:
                key, pkt = max(candidates, key=lambda item: receiver.last_arrival[item[0]])
                age = now - receiver.last_arrival[key]
                states[module] = dict(fresh=1, track_id=pkt.track_id, age_ms=age,
                    range_ft=f'{pkt.range_mm / 1000 * 3.28084:.1f}' if pkt.flags & RANGE_VALID else '',
                    closing_mph=f'{max(0, -pkt.radial_mmps / 1000 * 2.23694):.1f}' if pkt.flags & VELOCITY_VALID else '',
                    source='VIDEO TRACK' if module == Module.FRONT else 'SYNTHETIC RADAR')
            else:
                states[module] = dict(fresh=0, track_id='', age_ms='',
                                      range_ft='', closing_mph='', source='NO FRESH DATA')
        row = {'time_ms': now}
        for module, prefix in ((Module.FRONT, 'front'), (Module.REAR, 'rear')):
            row.update({f'{prefix}_{k}': v for k, v in states[module].items()})
        timeline.append(row)
    assert counters['sent'] == counters['accepted'] + counters['rejected'] + counters['dropped']
    assert all(not int(row['rear_fresh']) for row in timeline if 9300 <= row['time_ms'] < 13000)
    assert all(not row['front_range_ft'] and not row['front_closing_mph'] for row in timeline)
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / 'packet_log.csv', log, ['arrival_ms', 'source', 'result', 'packet_hex'])
    write_csv(OUT / 'cab_timeline.csv', timeline, list(timeline[0]))
    print('========== INTEGRATED VIDEO-TO-CAB PIPELINE ==========')
    print(f'Video tracking observations read: {observations} | 100-ms ticks with tracks: {len(ticks)}')
    print(f"Packets sent={counters['sent']} accepted={counters['accepted']} dropped={counters['dropped']} rejected={counters['rejected']}")
    print('Rear outage detection: PASS | Front metric range withheld: PASS')
    print(f'Results: {OUT}')
    print('Front: real video track IDs, NOT calibrated range/velocity')
    print('Rear: synthetic telemetry | No safety warnings authorized')
    return timeline


def display(speed=1.0):
    import tkinter as tk
    from tkinter import ttk
    with (OUT / 'cab_timeline.csv').open(newline='', encoding='utf-8') as f:
        timeline = list(csv.DictReader(f))
    if not timeline:
        raise ValueError('No timeline; run without --display first')
    root = tk.Tk()
    root.title('Vector Dynamics | Integrated Cab Prototype')
    root.geometry('820x420')
    root.configure(bg='#111827')
    fg, muted, panel = '#f9fafb', '#9ca3af', '#1f2937'
    tk.Label(root, text='VECTOR DYNAMICS  |  INTEGRATED VIDEO REPLAY',
             bg='#111827', fg=fg, font=('Segoe UI', 17, 'bold')).pack(pady=(20, 5))
    tk.Label(root, text='FRONT RANGE NOT MEASURED  ·  REAR SYNTHETIC  ·  NO SAFETY WARNINGS',
             bg='#111827', fg='#fbbf24').pack(pady=(0, 15))
    clock = tk.StringVar()
    tk.Label(root, textvariable=clock, bg='#111827', fg=muted).pack()
    cards = tk.Frame(root, bg='#111827')
    cards.pack(fill='both', expand=True, padx=20, pady=15)
    variables = {}
    for col, name in enumerate(('front', 'rear')):
        frame = tk.Frame(cards, bg=panel, padx=18, pady=14)
        frame.grid(row=0, column=col, padx=7, sticky='nsew')
        tk.Label(frame, text=name.upper(), bg=panel, fg=fg,
                 font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        vars_ = [tk.StringVar() for _ in range(4)]
        for i, var in enumerate(vars_):
            tk.Label(frame, textvariable=var, bg=panel,
                     fg=fg if i < 2 else muted,
                     font=('Segoe UI', 15 if i == 0 else 11)).pack(anchor='w', pady=4)
        variables[name] = vars_
    cards.grid_columnconfigure(0, weight=1, uniform='card')
    cards.grid_columnconfigure(1, weight=1, uniform='card')
    cards.grid_rowconfigure(0, weight=1)
    progress = ttk.Progressbar(root, maximum=len(timeline))
    progress.pack(fill='x', padx=26, pady=(0, 12))
    index = [0]
    def tick():
        if index[0] >= len(timeline):
            return
        row = timeline[index[0]]
        clock.set(f"Replay time: {int(row['time_ms']) / 1000:.2f} s")
        for name, vars_ in variables.items():
            fresh = int(row[f'{name}_fresh'])
            vars_[0].set(f"Track ID: {row[f'{name}_track_id']}" if fresh else 'NO FRESH DATA')
            vars_[1].set(f"Distance: {row[f'{name}_range_ft']} ft" if row[f'{name}_range_ft'] else 'Distance: unavailable')
            vars_[2].set(f"Closing: {row[f'{name}_closing_mph']} mph" if row[f'{name}_closing_mph'] else 'Closing: unavailable')
            vars_[3].set(f"{row[f'{name}_source']}  |  Age: {row[f'{name}_age_ms'] or '—'} ms")
        index[0] += 1
        progress['value'] = index[0]
        root.after(max(1, round(50 / speed)), tick)
    tick()
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, default=DEFAULT_TRACKS)
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--display', action='store_true', help='Show saved integrated timeline')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error('--speed must be positive')
    if args.display:
        display(args.speed)
    else:
        run(args.tracks, args.duration, args.seed)
