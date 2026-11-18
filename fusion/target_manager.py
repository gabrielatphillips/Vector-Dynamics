"""Vector Dynamics Phase 13: persistent simulated radar-camera target manager.

Place in fusion/phase13_target_manager.py beside radar_association.py (Phase 12).
Uses Phase 12's deterministic simulated scenes; no physical sensor measurements.
"""
from pathlib import Path
import csv
import math
import random
from collections import Counter

import radar_association as sim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fusion' / 'phase13_results'
OUT.mkdir(parents=True, exist_ok=True)

CONFIRM_HITS = 3
COAST_AFTER_S = 0.12
EXPIRE_AFTER_S = 0.65
MAX_ASSOC_AGE_S = 0.20
RADAR_GATE_DEG = 3.5
RADAR_GATE_RANGE_M = 8.0
RADAR_GATE_VELOCITY_MPS = 4.0


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def nearest_radar_track(r, tracks, now, used):
    options = []
    for tid, tr in tracks.items():
        if tid in used or tr['camera_id'] is not None:
            continue
        age = now - tr['last_seen']
        if age < -0.001 or age > EXPIRE_AFTER_S:
            continue
        predicted_range = tr['range_m'] + tr['velocity_mps'] * (r['time_s'] - tr['radar_time'])
        da = abs(tr['angle_deg'] - r['angle_deg'])
        dr = abs(predicted_range - r['range_m'])
        dv = abs(tr['velocity_mps'] - r['velocity_mps'])
        if da <= RADAR_GATE_DEG and dr <= RADAR_GATE_RANGE_M and dv <= RADAR_GATE_VELOCITY_MPS:
            options.append((da / RADAR_GATE_DEG + dr / RADAR_GATE_RANGE_M + dv / RADAR_GATE_VELOCITY_MPS, tid))
    return min(options)[1] if options else None


def main():
    if not sim.INPUT.exists():
        raise FileNotFoundError(f'Phase 12 input not found: {sim.INPUT}')
    rng = random.Random(sim.SEED)
    groups = sim.load_camera()
    world = sim.make_world(groups, rng)
    scenes, _, _ = sim.make_scenes(groups, world, rng)
    if not scenes:
        raise ValueError('No scenes to process')

    tracks = {}
    camera_to_fused = {}
    next_id = 1
    association_history = {}
    snapshots = []
    events = []
    counters = Counter()

    def new_track(now, camera_id=None):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks[tid] = {
            'id': tid, 'camera_id': camera_id, 'hits': 0,
            'last_seen': now, 'camera_time': None, 'radar_time': None,
            'range_m': None, 'velocity_mps': None, 'angle_deg': None,
            'source': 'none', 'status': 'tentative',
        }
        if camera_id is not None:
            camera_to_fused[camera_id] = tid
        counters['created'] += 1
        return tid

    for frame, t, cams, radar in scenes:
        # Phase 12 global matching; this history is separate from track lifecycle.
        pairs = sim.global_assign(cams, radar, association_history, t)
        used_cam = set()
        used_rad = set()
        updated = set()
        used_radar_tracks = set()

        for ci, ri, cost in pairs:
            c, r = cams[ci], radar[ri]
            # Reject unusually asynchronous measurements.
            if abs(r['time_s'] - c['time_s']) > MAX_ASSOC_AGE_S:
                counters['time_rejected'] += 1
                continue
            used_cam.add(ci)
            used_rad.add(ri)
            cid = c['id']
            tid = camera_to_fused.get(cid)
            if tid is None or tid not in tracks:
                tid = new_track(t, cid)
            tr = tracks[tid]
            tr.update({
                'last_seen': t, 'camera_time': c['time_s'],
                'radar_time': r['time_s'], 'range_m': r['range_m'],
                'velocity_mps': r['velocity_mps'], 'angle_deg': r['angle_deg'],
                'source': 'fused',
            })
            tr['hits'] += 1
            updated.add(tid)
            counters['fused_updates'] += 1
            correct = r['truth_id'] is not None and cid == r['truth_id']
            counters['correct_pairs' if correct else 'incorrect_pairs'] += 1
            association_history[cid] = {
                'range_m': r['range_m'], 'velocity_mps': r['velocity_mps'],
                'time_s': r['time_s'],
            }
            events.append({'frame': frame, 'time_s': t, 'fused_id': tid,
                           'event': 'fused_update', 'camera_id': cid,
                           'radar_id': r['radar_id'], 'truth_id_debug': r['truth_id'],
                           'correct_match_debug': correct})

        # Camera-only updates: preserve identity but never invent radar distance.
        for ci, c in enumerate(cams):
            if ci in used_cam:
                continue
            cid = c['id']
            tid = camera_to_fused.get(cid)
            if tid is None or tid not in tracks:
                tid = new_track(t, cid)
            tr = tracks[tid]
            tr.update({'last_seen': t, 'camera_time': c['time_s'],
                       'source': 'camera_only'})
            tr['hits'] += 1
            updated.add(tid)
            counters['camera_only_updates'] += 1

        # Unmatched radar: use measured range, bearing and velocity for persistence.
        # Clutter can start tentative tracks; confirmation reduces one-off outputs.
        for ri, r in enumerate(radar):
            if ri in used_rad:
                continue
            tid = nearest_radar_track(r, tracks, t, used_radar_tracks)
            if tid is None:
                tid = new_track(t)
            tr = tracks[tid]
            tr.update({'last_seen': t, 'radar_time': r['time_s'],
                       'range_m': r['range_m'], 'velocity_mps': r['velocity_mps'],
                       'angle_deg': r['angle_deg'], 'source': 'radar_only'})
            tr['hits'] += 1
            updated.add(tid)
            used_radar_tracks.add(tid)
            counters['radar_only_updates'] += 1
            events.append({'frame': frame, 'time_s': t, 'fused_id': tid,
                           'event': 'radar_only_update', 'camera_id': '',
                           'radar_id': r['radar_id'], 'truth_id_debug': r['truth_id'],
                           'correct_match_debug': ''})

        # Lifecycle and per-frame output.
        for tid, tr in list(tracks.items()):
            age = max(0.0, t - tr['last_seen'])
            if age > EXPIRE_AFTER_S:
                counters['expired'] += 1
                events.append({'frame': frame, 'time_s': t, 'fused_id': tid,
                               'event': 'expired', 'camera_id': tr['camera_id'],
                               'radar_id': '', 'truth_id_debug': '',
                               'correct_match_debug': ''})
                if tr['camera_id'] is not None and camera_to_fused.get(tr['camera_id']) == tid:
                    del camera_to_fused[tr['camera_id']]
                del tracks[tid]
                continue

            if tid in updated:
                tr['status'] = 'confirmed' if tr['hits'] >= CONFIRM_HITS else 'tentative'
            elif age > COAST_AFTER_S:
                tr['status'] = 'coasting' if tr['hits'] >= CONFIRM_HITS else 'tentative'
            # Do not treat old range as a fresh measurement.
            radar_age_ms = (1000 * (t - tr['radar_time'])
                            if tr['radar_time'] is not None else None)
            radar_valid = (radar_age_ms is not None and
                           -MAX_ASSOC_AGE_S * 1000 <= radar_age_ms <= MAX_ASSOC_AGE_S * 1000)
            if tr['range_m'] is not None and radar_valid:
                estimated_range = tr['range_m'] + tr['velocity_mps'] * (t - tr['radar_time'])
            else:
                estimated_range = None
            camera_age_ms = (1000 * (t - tr['camera_time'])
                             if tr['camera_time'] is not None else None)
            camera_valid = (camera_age_ms is not None and
                            0 <= camera_age_ms <= MAX_ASSOC_AGE_S * 1000)
            # Source means sensors currently valid, not merely the last update type.
            source = ('fused' if camera_valid and radar_valid else
                      'camera_only' if camera_valid else
                      'radar_only' if radar_valid else 'none')
            snapshots.append({
                'frame': frame, 'time_s': round(t, 4), 'fused_id': tid,
                'camera_track_id': tr['camera_id'] if tr['camera_id'] is not None else '',
                'status': tr['status'], 'source': source,
                'camera_valid': camera_valid, 'radar_valid': radar_valid,
                'age_since_update_ms': round(age * 1000, 2),
                'camera_age_ms': round(camera_age_ms, 2) if camera_age_ms is not None else '',
                'radar_age_ms': round(radar_age_ms, 2) if radar_age_ms is not None else '',
                'range_m': round(estimated_range, 3) if estimated_range is not None else '',
                'range_ft': round(estimated_range * 3.28084, 2) if estimated_range is not None else '',
                'radial_velocity_mps': round(tr['velocity_mps'], 3) if radar_valid else '',
                'closing_speed_mph': round(max(0.0, -tr['velocity_mps']) * 2.23694, 2) if radar_valid else '',
                'angle_deg': round(tr['angle_deg'], 3) if radar_valid else '',
                'range_uncertainty_m': '',  # not estimated by this prototype
                'measurement_is_synthetic': True,
                'warning_authorized': False,
            })

    fields = list(snapshots[0]) if snapshots else []
    write_csv(OUT / 'target_snapshots.csv', snapshots, fields)
    write_csv(OUT / 'target_events.csv', events, list(events[0]) if events else
              ['frame', 'time_s', 'fused_id', 'event', 'camera_id', 'radar_id', 'truth_id_debug', 'correct_match_debug'])
    status_counts = Counter(r['status'] for r in snapshots)
    source_counts = Counter(r['source'] for r in snapshots)
    print('\n========== PHASE 13: TARGET MANAGER ==========')
    print(f'Frames: {len(scenes)} | Unique fused IDs created: {counters["created"]}')
    print(f'Fused updates: {counters["fused_updates"]} | Camera-only updates: {counters["camera_only_updates"]} | Radar-only updates: {counters["radar_only_updates"]}')
    print(f'Fused association correct: {counters["correct_pairs"]} | incorrect: {counters["incorrect_pairs"]} (synthetic truth only)')
    print(f'Expired tracks: {counters["expired"]} | Time-rejected associations: {counters["time_rejected"]}')
    print('Snapshot states:', dict(status_counts))
    print('Snapshot sensor sources:', dict(source_counts))
    print(f'Saved: {OUT}')
    print('NOTE: Synthetic radar only. No warnings authorized.')
    print('==============================================')

if __name__ == '__main__':
    main()
