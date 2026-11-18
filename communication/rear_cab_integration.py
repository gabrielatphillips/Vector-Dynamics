"""Vector Dynamics: front fused snapshots + rear selector CSV -> virtual cab.

Simulation only. Radar and lane geometry are synthetic; no safety decisions.
Run from VectorDynamics root:
  python communication/rear_cab_integration.py --self-test
  python communication/rear_cab_integration.py --display

The 32-byte protocol has no explicit NO_TARGET vs LANE_UNKNOWN reason code.
This desktop bridge uses the rear CSV reason as simulation-only sideband data.
Do NOT treat that distinction as available over ESP-NOW without a protocol update.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from vehicle_protocol import (Module, Status, Receiver, decode, encode, SYNTHETIC,
                              RANGE_VALID, VELOCITY_VALID)
from fused_cab_pipeline import (SNAPSHOTS, load_snapshots, packet_from_snapshot)

REAR_CSV = ROOT / 'fusion' / 'rear_detection_results' / 'rear_selection_timeline.csv'
OUT = HERE / 'rear_cab_results'
COLUMNS = ['time_ms', 'front_state', 'front_track_id', 'front_range_ft',
           'front_closing_mph', 'front_age_ms', 'rear_state', 'rear_track_id',
           'rear_range_ft', 'rear_closing_mph', 'rear_age_ms']


def read_rear(path):
    rows = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'time_ms', 'valid', 'reason', 'packet_hex'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Rear CSV missing columns: {required - set(reader.fieldnames or [])}')
        for row in reader:
            ms = int(row['time_ms'])
            if ms in rows:
                raise ValueError(f'Duplicate rear timestamp {ms}')
            raw = bytes.fromhex(row['packet_hex'])
            pkt = decode(raw)
            if pkt.source != Module.REAR or pkt.destination != Module.CAB:
                raise ValueError(f'Unexpected rear packet routing at {ms} ms')
            if pkt.sender_ms != ms:
                raise ValueError(f'Rear timestamp mismatch at {ms} ms')
            valid = row['valid'].strip() == '1'
            if valid != bool(pkt.flags & RANGE_VALID and pkt.flags & VELOCITY_VALID):
                raise ValueError(f'Rear validity/packet flag mismatch at {ms} ms')
            if not pkt.flags & SYNTHETIC:
                raise ValueError('This demo requires explicitly synthetic packets')
            reason = row['reason'].strip().lower()
            if valid:
                state = 'TRACKED'
            elif ('lane geometry' in reason or 'uncertain' in reason or 'ambiguous' in reason or 'closer possible' in reason):
                state = 'LANE UNKNOWN'
            else:
                state = 'NO TARGET'
            rows[ms] = (raw, state)
    if not rows:
        raise ValueError('Rear CSV is empty')
    return rows


def simulate(rear_csv=REAR_CSV, snapshots=SNAPSHOTS, seed=42,
             drop_rate=.10, corrupt_rate=.04, freshness_ms=300,
             rear_outage_start=9000, rear_outage_end=13000):
    if not (0 <= drop_rate <= 1 and 0 <= corrupt_rate <= 1 and freshness_ms > 0):
        raise ValueError('Invalid simulation parameters')
    rear = read_rear(rear_csv)
    duration_ms = max(rear) + 100
    front, _ = load_snapshots(Path(snapshots), duration_ms / 1000)
    rng = random.Random(seed)
    receiver = Receiver(max_age_ms=freshness_ms)
    # One current message per source, even if the track ID changes. This prevents
    # a previous track from reappearing after a NO TARGET / UNKNOWN packet.
    latest = {}
    pending = []
    log = []
    timeline = []
    counts = Counter()
    serial = 0
    front_seq = 0
    for ms in range(0, duration_ms, 100):
        messages = []
        choices = front.get(ms // 100, [])
        if choices:
            def priority(r):
                radar = r['radar_valid'].lower() in ('true', '1') and bool(r['range_m'])
                camera = r['camera_valid'].lower() in ('true', '1')
                return (2 if radar and camera else 1 if radar else 0,
                        -float(r['range_m']) if radar else -9999, -int(r['fused_id']))
            pkt = packet_from_snapshot(max(choices, key=priority), front_seq, ms)
            messages.append((encode(pkt), 'TRACKED'))
        front_seq += 1
        if ms in rear:
            if rear_outage_start <= ms < rear_outage_end:
                counts['rear_disconnected'] += 1
            else:
                messages.append(rear[ms])
        for raw, state in messages:
            counts['sent'] += 1
            if rng.random() < drop_rate:
                counts['dropped'] += 1
                continue
            data = bytearray(raw)
            if rng.random() < corrupt_rate:
                data[rng.randrange(len(data)-4)] ^= 1
                counts['corrupted'] += 1
            delivery = ms + rng.randint(20, 120)
            heapq.heappush(pending, (delivery, serial, bytes(data), state))
            serial += 1
    for now in range(0, duration_ms + freshness_ms + 201, 50):
        while pending and pending[0][0] <= now:
            arrival, _, raw, state = heapq.heappop(pending)
            try:
                pkt = receiver.ingest(raw, arrival)
            except ValueError:
                counts['rejected'] += 1
                log.append({'arrival_ms': arrival, 'source': 'unknown', 'result': 'rejected', 'state': '', 'packet_hex': raw.hex()})
                continue
            counts['accepted'] += 1
            # Sideband state is simulation metadata, not present in the binary packet.
            latest[pkt.source] = (pkt, arrival, state)
            log.append({'arrival_ms': arrival, 'source': pkt.source.name.lower(), 'result': 'accepted', 'state': state, 'packet_hex': raw.hex()})
        item = {'time_ms': now}
        for module, name in ((Module.FRONT, 'front'), (Module.REAR, 'rear')):
            current = latest.get(module)
            if not current or now - current[1] > freshness_ms:
                item.update({f'{name}_state': 'SIGNAL LOST', f'{name}_track_id': '',
                             f'{name}_range_ft': '', f'{name}_closing_mph': '', f'{name}_age_ms': ''})
                continue
            pkt, arrival, state = current
            valid = state == 'TRACKED' and pkt.status == Status.CONFIRMED
            item.update({f'{name}_state': state,
                         f'{name}_track_id': pkt.track_id if valid else '',
                         f'{name}_range_ft': f'{pkt.range_m * 3.28084:.1f}' if valid and pkt.range_m is not None else '',
                         f'{name}_closing_mph': f'{max(0, -pkt.radial_mps * 2.23694):.1f}' if valid and pkt.radial_mps is not None else '',
                         f'{name}_age_ms': now - arrival})
        timeline.append(item)
    assert counts['sent'] == counts['accepted'] + counts['rejected'] + counts['dropped']
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'cab_timeline.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerows(timeline)
    with (OUT / 'packet_log.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arrival_ms','source','result','state','packet_hex'])
        writer.writeheader(); writer.writerows(log)
    states = Counter(row['rear_state'] for row in timeline if row['time_ms'] < duration_ms)
    print('========== REAR-TO-CAB INTEGRATION ==========')
    print(f'Rear input records: {len(rear)} | Sent: {counts["sent"]} | Accepted: {counts["accepted"]} | Dropped: {counts["dropped"]} | Rejected: {counts["rejected"]}')
    print('Rear cab states:', dict(states))
    print('Results:', OUT)
    print('Synthetic radar/lane geometry. NO safety warnings authorized.')
    return timeline


def display(speed=1.0):
    import tkinter as tk
    if speed <= 0:
        raise ValueError('Speed must be positive')
    with (OUT / 'cab_timeline.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    root = tk.Tk()
    root.title('Vector Dynamics | Front + Rear Cab Replay')
    root.geometry('860x370')
    root.configure(bg='#111827')
    tk.Label(root, text='VECTOR DYNAMICS  |  FRONT + REAR REPLAY',
             bg='#111827', fg='white', font=('Segoe UI', 16, 'bold')).pack(pady=(18, 4))
    tk.Label(root, text='SYNTHETIC RADAR AND LANE DATA — NO SAFETY WARNINGS',
             bg='#111827', fg='#fbbf24').pack()
    clock = tk.StringVar()
    tk.Label(root, textvariable=clock, bg='#111827', fg='#9ca3af').pack(pady=10)
    body = tk.Frame(root, bg='#111827'); body.pack(fill='both', expand=True, padx=20)
    widgets = {}
    for col, name in enumerate(('front', 'rear')):
        card = tk.Frame(body, bg='#1f2937', padx=18, pady=14)
        card.grid(row=0, column=col, sticky='nsew', padx=6)
        tk.Label(card, text=name.upper(), bg='#1f2937', fg='white', font=('Segoe UI', 15, 'bold')).pack(anchor='w')
        values = [tk.StringVar() for _ in range(4)]
        for value in values:
            tk.Label(card, textvariable=value, bg='#1f2937', fg='white', font=('Segoe UI', 12)).pack(anchor='w', pady=5)
        widgets[name] = values
    for col in (0, 1): body.grid_columnconfigure(col, weight=1, uniform='panel')
    body.grid_rowconfigure(0, weight=1)
    i = [0]
    def tick():
        if i[0] >= len(rows): return
        row = rows[i[0]]
        clock.set(f'Time: {int(row["time_ms"]) / 1000:.2f} s')
        for name, values in widgets.items():
            state = row[f'{name}_state']
            values[0].set(f'Status: {state}')
            values[1].set(f'Track: {row[f"{name}_track_id"] or "—"}')
            values[2].set(f'Distance: {row[f"{name}_range_ft"] + " ft" if row[f"{name}_range_ft"] else "—"}')
            values[3].set(f'Closing: {row[f"{name}_closing_mph"] + " mph" if row[f"{name}_closing_mph"] else "—"}')
        i[0] += 1
        root.after(max(1, round(50 / speed)), tick)
    tick(); root.mainloop()


def self_test():
    # Test states using the supplied 150-row synthetic timeline, without network faults.
    rows = simulate(drop_rate=0, corrupt_rate=0, rear_outage_start=20000, rear_outage_end=20000)
    seen = {row['rear_state'] for row in rows}
    assert {'TRACKED', 'LANE UNKNOWN', 'NO TARGET', 'SIGNAL LOST'} <= seen, seen
    for row in rows:
        if row['rear_state'] != 'TRACKED':
            assert row['rear_range_ft'] == '' and row['rear_closing_mph'] == ''
    # Test the outage independently.
    outage = simulate(drop_rate=0, corrupt_rate=0, rear_outage_start=9000, rear_outage_end=13000)
    assert all(row['rear_state'] == 'SIGNAL LOST' for row in outage if 9400 <= row['time_ms'] < 13000)
    print('Rear-to-cab state and outage checks: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rear-csv', type=Path, default=REAR_CSV)
    parser.add_argument('--snapshots', type=Path, default=SNAPSHOTS)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        simulate(args.rear_csv, args.snapshots)
        if args.display: display(args.speed)
