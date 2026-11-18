"""Vector Dynamics: fused snapshot -> rear lane selector -> protocol replay.

Input snapshots MUST originate from a rear-facing sensor/fusion pipeline. The
existing phase13 target manager processes FRONT video; its output must not be
silently treated as rear data. Use --synthetic-rear-replay only for software tests.
No real lane detector, radar calibration, or safety warnings are implemented.
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'fusion', ROOT / 'tracking', ROOT / 'communication'):
    sys.path.insert(0, str(folder))
from rear_lane_estimator import LaneEstimate
from rear_detection_pipeline import RearDetectionPipeline, Observation
from vehicle_protocol import encode, decode

FIELDS = ['time_ms','selected_id','status','range_m','closing_mps','valid','packet_hex','reason',
          'source','candidate_count','classifications','measurement_is_synthetic','warning_authorized']


def truthy(value):
    return str(value).strip().lower() in ('true','1','yes')


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def convert(row, now_ms, *, bearing_std_deg=0.2, range_std_m=0.2):
    """Only confirmed, radar-valid, sufficiently fresh fused measurements."""
    if row.get('status','').strip().lower() != 'confirmed':
        return None
    if not truthy(row.get('radar_valid','false')):
        return None
    if not truthy(row.get('measurement_is_synthetic','true')):
        # This bridge is not calibrated for real sensor measurements.
        raise ValueError('Real sensor measurements require calibrated integration; refusing replay')
    age = number(row.get('radar_age_ms', '0'))
    if age is None or age < -200 or age > 200:
        return None
    r = number(row.get('range_m'))
    v = number(row.get('radial_velocity_mps'))
    a = number(row.get('angle_deg'))
    if r is None or v is None or a is None or r <= 0:
        return None
    try:
        tid = int(row['fused_id'])
    except (ValueError, KeyError):
        return None
    # Snapshot time is the observation time only when radar age is 0; keep
    # the actual measurement timestamp so coasting cannot masquerade as fresh.
    measured_ms = now_ms - max(0, int(round(age)))
    return Observation(tid, r, a, -v, measured_ms, True, bearing_std_deg, range_std_m)


def run(input_path: Path, output_path: Path, *, synthetic_rear_replay=False,
        lane_width_m=3.7, lane_confidence=0.98, lane_center_m=0.0,
        lane_std_m=0.05, bearing_std_deg=0.2, range_std_m=0.2):
    if not synthetic_rear_replay:
        raise ValueError('Rear-facing provenance not verified. For a synthetic test use --synthetic-rear-replay.')
    groups = defaultdict(list)
    with input_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'time_s','fused_id','status','radar_valid','range_m','radial_velocity_mps','angle_deg'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Missing columns: {sorted(required-set(reader.fieldnames or []))}')
        for row in reader:
            t = number(row['time_s'])
            if t is None or t < 0:
                raise ValueError('Invalid time_s')
            groups[round(t*1000)].append(row)
    if not groups:
        raise ValueError('No snapshot rows')
    pipeline = RearDetectionPipeline()
    results = []
    counts = Counter()
    for now_ms in sorted(groups):
        observations = []
        for row in groups[now_ms]:
            obs = convert(row, now_ms, bearing_std_deg=bearing_std_deg, range_std_m=range_std_m)
            if obs is not None:
                observations.append(obs)
        # This lane is intentionally a constant synthetic geometry fixture;
        # a real rear-facing lane estimator must supply a per-range lane model.
        lane = LaneEstimate(lane_center_m, lane_width_m, lane_std_m, now_ms, lane_confidence)
        packet, meta = pipeline.step(observations, lane, now_ms)
        raw = encode(packet)
        assert len(raw) == 32
        decoded = decode(raw)
        assert decoded is not None
        status = meta['status']
        counts[status] += 1
        results.append({'time_ms':now_ms, 'selected_id':meta['selected_id'] or '',
                        'status':status,'range_m':meta['range_m'] if meta['range_m'] is not None else '',
                        'closing_mps':meta['closing_mps'] if meta['closing_mps'] is not None else '',
                        'valid':int(status=='tracked'), 'packet_hex':raw.hex(),
                        'reason':meta['reason'], 'source':'synthetic_rear_replay',
                        'candidate_count':len(observations),
                        'classifications':str(meta['classifications']),
                        'measurement_is_synthetic':True,'warning_authorized':False})
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(results)
    print(f'Rear fused snapshot groups: {len(groups)} | Selection states: {dict(counts)}')
    print(f'Saved: {output_path}')
    print('SYNTHETIC rear replay, constant lane geometry. NO safety warnings authorized.')
    return results


def self_test():
    import tempfile
    columns=['time_s','fused_id','status','radar_valid','radar_age_ms','range_m',
             'radial_velocity_mps','angle_deg','measurement_is_synthetic']
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'snapshots.csv'; out=Path(d)/'rear_selection_timeline.csv'
        rows=[]
        def add(t,tid,r,a,status='confirmed',radar_valid='true',age='0'):
            rows.append(dict(time_s=t,fused_id=tid,status=status,radar_valid=radar_valid,
                             radar_age_ms=age,range_m=r,radial_velocity_mps=-3,
                             angle_deg=a,measurement_is_synthetic='true'))
        # 10 m adjacent vehicle, 20 m in-lane vehicle.
        add(1.0,1,20,0); add(1.0,2,10,22)
        add(1.1,1,20,0); add(1.1,2,10,22)
        add(1.2,1,20,0); add(1.2,2,10,22)
        # Boundary ambiguity blocks closer target.
        add(1.3,3,10,9.8); add(1.3,1,20,0)
        # No confirmed candidates -> unknown.
        add(1.4,1,20,0,status='tentative')
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
        result=run(p,out,synthetic_rear_replay=True)
        assert len(result)==5
        assert all(x['selected_id']==1 for x in result[:3]),result
        assert all(x['valid']==1 for x in result[:3]),result
        assert result[3]['valid']==0 and result[4]['valid']==0,result
        assert all(len(bytes.fromhex(x['packet_hex']))==32 for x in result)
        assert all(x['warning_authorized'] is False for x in result)
        try:
            run(p,out)
        except ValueError:
            pass
        else:
            raise AssertionError('Provenance guard failed')
    print('Rear live integration: 7 checks PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'fusion'/'phase13_results'/'target_snapshots.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'fusion'/'rear_live_results'/'rear_selection_timeline.csv')
    parser.add_argument('--synthetic-rear-replay',action='store_true',help='Explicitly allow synthetic replay of snapshots (NOT a physical rear sensor)')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run(args.input,args.output,synthetic_rear_replay=args.synthetic_rear_replay)

if __name__=='__main__':main()
