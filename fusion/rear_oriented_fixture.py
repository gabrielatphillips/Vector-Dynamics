"""Deterministic REAR-ORIENTED synthetic fused snapshots for integration testing.

This is NOT a rear-camera detector or a real radar capture. Coordinates are
explicitly synthetic, with bearing 0 degrees pointing directly behind the truck.
Run from project root: python fusion/rear_oriented_fixture.py --self-test
Then: python fusion/rear_oriented_fixture.py
"""
from __future__ import annotations
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fusion' / 'rear_oriented_results'
COLUMNS = ['time_s','fused_id','status','source','radar_valid','camera_valid',
           'radar_age_ms','range_m','radial_velocity_mps','angle_deg','measurement_is_synthetic']


def make_rows(frames=150):
    if frames < 100:
        raise ValueError('At least 100 frames needed for all test scenarios')
    rows = []
    for frame in range(frames):
        t = frame / 10.0
        # 0-49: target 1 directly behind; closer target 2 in adjacent lane.
        # 50-69: a closer lane-boundary target 3 creates ambiguity.
        # 70-89: no confirmed in-lane vehicle.
        # 90-119: target 4 follows in lane; closer adjacent target 2 excluded.
        # 120+: no eligible target (unconfirmed / camera-only).
        if frame < 50:
            objects = [(1, 24 - .05*frame, 0, 'confirmed', True),
                       (2, 12, 20, 'confirmed', True)]
        elif frame < 70:
            objects = [(1, 21, 0, 'confirmed', True),
                       (3, 10, 9.8, 'confirmed', True)]
        elif frame < 90:
            objects = [(2, 12, 20, 'confirmed', True)]
        elif frame < 120:
            objects = [(4, 30 - .04*(frame-90), 0, 'confirmed', True),
                       (2, 12, 20, 'confirmed', True)]
        else:
            objects = [(4, 29, 0, 'tentative', True),
                       (2, 12, 20, 'confirmed', True)]
        for tid, distance, angle, status, radar_valid in objects:
            rows.append(dict(time_s=f'{t:.1f}', fused_id=tid, status=status,
                             source='fused' if radar_valid else 'camera_only',
                             radar_valid=str(radar_valid).lower(),camera_valid='true',
                             radar_age_ms=0,range_m=f'{distance:.3f}',
                             radial_velocity_mps='-2.5',angle_deg=angle,
                             measurement_is_synthetic='true'))
    return rows


def build(output=OUT, frames=150):
    import sys
    sys.path.insert(0, str(ROOT / 'fusion'))
    from rear_live_integration import run
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / 'rear_fused_snapshots.csv'
    timeline = output / 'rear_selection_timeline.csv'
    with snapshots.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(make_rows(frames))
    selected = run(snapshots,timeline,synthetic_rear_replay=True)
    counts = Counter(r['status'] for r in selected)
    print(f'Generated {frames} rear-oriented synthetic frames | {dict(counts)}')
    print(f'Rear snapshots: {snapshots}\nCab-compatible rear timeline: {timeline}')
    return selected


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rows=build(Path(tmp))
        assert len(rows)==150
        assert all(r['selected_id']==1 and r['valid']==1 for r in rows[:50])
        assert all(r['valid']==0 for r in rows[50:90])
        assert all(r['selected_id']==4 and r['valid']==1 for r in rows[90:120])
        assert all(r['valid']==0 for r in rows[120:])
        assert all(len(bytes.fromhex(r['packet_hex']))==32 for r in rows)
        assert all(r['warning_authorized'] is False for r in rows)
    print('Rear-oriented fixture: 6 checks PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--frames',type=int,default=150)
    args=p.parse_args()
    if args.self_test: self_test()
    else: build(frames=args.frames)
