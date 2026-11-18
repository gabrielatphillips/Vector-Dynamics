"""Vector Dynamics: conservative persistent track manager experiment.

Synthetic evaluation only. Does not authorize driving warnings.
Requires radar_association.py and track_reconciliation.py in the fusion folder.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import random
import radar_association as sim
import track_reconciliation as baseline
from track_reconciliation import (radar_cost, write_csv, CONFIRM_HITS,
    COAST_AFTER_S, EXPIRE_AFTER_S, MAX_ASSOC_AGE_S,
    RADAR_GATE_DEG, RADAR_GATE_RANGE_M, RADAR_GATE_VELOCITY_MPS)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fusion' / 'stable_track_results'
OUT.mkdir(parents=True, exist_ok=True)

# Conservative experimental thresholds, NOT safety-certified.
ADOPTION_MIN_HITS = 3
ADOPTION_MAX_AGE_S = 0.16
ADOPTION_ANGLE_DEG = 1.5
ADOPTION_RANGE_M = 3.0
ADOPTION_VELOCITY_MPS = 1.5
ADOPTION_MAX_COST = 1.6
ADOPTION_MARGIN = 0.5

def run_stable(scenes):
    tracks = {}
    camera_to_id = {}
    history = {}
    next_id = 1
    events = []
    snapshots = []
    counters = Counter()
    truth_ids_by_track = defaultdict(set)  # EVALUATION ONLY
    truth_to_track_ids = defaultdict(set)  # EVALUATION ONLY
    truth_active_history = defaultdict(list)

    def create(t, camera_id=None):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks[tid] = dict(id=tid, camera_id=camera_id, hits=0,
                           last_seen=t, camera_time=None, radar_time=None,
                           range_m=None, velocity_mps=None, angle_deg=None,
                           status='tentative')
        if camera_id is not None:
            camera_to_id[camera_id] = tid
        counters['created'] += 1
        return tid

    def update_radar(tid, r, t, camera=None):
        tr = tracks[tid]
        tr.update(last_seen=t, radar_time=r['time_s'], range_m=r['range_m'],
                  velocity_mps=r['velocity_mps'], angle_deg=r['angle_deg'])
        if camera is not None:
            tr['camera_time'] = camera['time_s']
        tr['hits'] += 1
        # Ground truth is recorded AFTER association and never used for matching.
        truth = r['truth_id']
        if truth is not None:
            truth_ids_by_track[tid].add(truth)
            truth_to_track_ids[truth].add(tid)
        return tr

    for frame, t, cams, radar in scenes:
        # Expire before association so old tracks cannot be resurrected.
        for tid, tr in list(tracks.items()):
            if t - tr['last_seen'] > EXPIRE_AFTER_S:
                counters['expired'] += 1
                events.append(dict(frame=frame, time_s=t, event='expired',
                                   track_id=tid, other_track_id='', camera_id='',
                                   radar_id='', truth_debug=''))
                if tr['camera_id'] is not None and camera_to_id.get(tr['camera_id']) == tid:
                    del camera_to_id[tr['camera_id']]
                del tracks[tid]

        pairs = sim.global_assign(cams, radar, history, t)
        used_cam, used_rad, updated = set(), set(), set()
        for ci, ri, _ in pairs:
            c, r = cams[ci], radar[ri]
            if abs(c['time_s'] - r['time_s']) > MAX_ASSOC_AGE_S:
                counters['time_rejected'] += 1
                continue
            used_cam.add(ci)
            used_rad.add(ri)
            cid = c['id']
            tid = camera_to_id.get(cid)
            if tid not in tracks:
                tid = None
            # Reconcile a previously radar-only track with the new camera track.
            # Only geometric/kinematic measurements are used, NEVER truth_id.
            # Adopt only an established radar-only track; never merge two IDs.
            # Association uses measurements, not synthetic truth.
            if tid is None:
                candidates = []
                for rid, old in tracks.items():
                    if old['camera_id'] is not None or rid in updated:
                        continue
                    if old['hits'] < ADOPTION_MIN_HITS:
                        continue
                    if old['radar_time'] is None or abs(r['time_s'] - old['radar_time']) > ADOPTION_MAX_AGE_S:
                        continue
                    score = radar_cost(old, r, ADOPTION_ANGLE_DEG,
                                       ADOPTION_RANGE_M, ADOPTION_VELOCITY_MPS)
                    if score is not None:
                        candidates.append((score, rid))
                candidates.sort()
                if candidates:
                    best_score, best_id = candidates[0]
                    runner_up = candidates[1][0] if len(candidates) > 1 else float('inf')
                    if best_score <= ADOPTION_MAX_COST and runner_up - best_score >= ADOPTION_MARGIN:
                        tid = best_id
                        old_truth = sorted(truth_ids_by_track[tid])
                        tracks[tid]['camera_id'] = cid
                        camera_to_id[cid] = tid
                        counters['radar_to_camera_adoptions'] += 1
                        events.append(dict(frame=frame, time_s=t, event='radar_to_camera_adoption',
                                           track_id=tid, other_track_id='', camera_id=cid,
                                           radar_id=r['radar_id'], truth_debug=r['truth_id'],
                                           donor_truth_history=';'.join(map(str, old_truth)),
                                           receiver_truth_history=str(r['truth_id'])))
                    else:
                        counters['ambiguous_adoptions_rejected'] += 1
            if tid is None:
                tid = create(t, cid)
            tr = tracks[tid]
            tr['camera_id'] = cid
            camera_to_id[cid] = tid
            update_radar(tid, r, t, camera=c)
            updated.add(tid)
            counters['fused_updates'] += 1
            correct = r['truth_id'] is not None and cid == r['truth_id']
            counters['correct_pairs' if correct else 'incorrect_pairs'] += 1
            history[cid] = dict(range_m=r['range_m'], velocity_mps=r['velocity_mps'],
                                time_s=r['time_s'])

        for ci, c in enumerate(cams):
            if ci in used_cam:
                continue
            cid = c['id']
            tid = camera_to_id.get(cid)
            if tid not in tracks:
                tid = create(t, cid)
            tr = tracks[tid]
            tr['last_seen'] = t
            tr['camera_time'] = c['time_s']
            tr['hits'] += 1
            updated.add(tid)
            counters['camera_only_updates'] += 1

        # One-to-one radar-only association to persistent tracks.
        for ri, r in enumerate(radar):
            if ri in used_rad:
                continue
            options = []
            for tid, tr in tracks.items():
                if tid in updated or tr['camera_id'] is not None:
                    continue
                score = radar_cost(tr, r, RADAR_GATE_DEG,
                                   RADAR_GATE_RANGE_M, RADAR_GATE_VELOCITY_MPS)
                if score is not None:
                    options.append((score, tid))
            tid = min(options)[1] if options else create(t)
            update_radar(tid, r, t)
            updated.add(tid)
            counters['radar_only_updates'] += 1

        # Track-state snapshots and synthetic-truth diagnostics.
        current_truth_owners = defaultdict(set)
        for tid, tr in tracks.items():
            age = max(0, t - tr['last_seen'])
            if tid in updated:
                tr['status'] = 'confirmed' if tr['hits'] >= CONFIRM_HITS else 'tentative'
            elif age > COAST_AFTER_S:
                tr['status'] = 'coasting' if tr['hits'] >= CONFIRM_HITS else 'tentative'
            ca = None if tr['camera_time'] is None else t - tr['camera_time']
            ra = None if tr['radar_time'] is None else t - tr['radar_time']
            camera_valid = ca is not None and 0 <= ca <= MAX_ASSOC_AGE_S
            radar_valid = ra is not None and -MAX_ASSOC_AGE_S <= ra <= MAX_ASSOC_AGE_S
            source = ('fused' if camera_valid and radar_valid else
                      'camera_only' if camera_valid else
                      'radar_only' if radar_valid else 'none')
            range_m = (tr['range_m'] + tr['velocity_mps'] * ra
                       if radar_valid and tr['range_m'] is not None else None)
            snapshots.append(dict(frame=frame, time_s=round(t, 4), fused_id=tid,
                                  camera_track_id=tr['camera_id'] if tr['camera_id'] is not None else '',
                                  status=tr['status'], source=source,
                                  range_m=round(range_m, 3) if range_m is not None else '',
                                  radial_velocity_mps=round(tr['velocity_mps'], 3) if radar_valid else '',
                                  camera_age_ms=round(1000*ca, 2) if ca is not None else '',
                                  radar_age_ms=round(1000*ra, 2) if ra is not None else '',
                                  measurement_is_synthetic=True, warning_authorized=False))
            # This is a conservative diagnostic: all truth IDs ever attached to
            # this track are counted, so it can overstate simultaneous duplicates.
            for truth in truth_ids_by_track[tid]:
                current_truth_owners[truth].add(tid)
        counters['duplicate_truth_frames'] += sum(len(ids) > 1 for ids in current_truth_owners.values())
        for truth, ids in current_truth_owners.items():
            truth_active_history[truth].append((frame, tuple(sorted(ids))))

    fragmentation = sum(max(0, len(ids) - 1) for ids in truth_to_track_ids.values())
    contamination = sum(len(ids) > 1 for ids in truth_ids_by_track.values())
    confirmed_clutter = sum(tr['hits'] >= CONFIRM_HITS and not truth_ids_by_track[tid]
                            for tid, tr in tracks.items())
    counters['fragmentation_extra_ids'] = fragmentation
    counters['mixed_truth_tracks'] = contamination
    counters['confirmed_clutter_active_at_end'] = confirmed_clutter
    return counters, snapshots, events


def main():
    if not sim.INPUT.exists():
        raise FileNotFoundError(f'Missing camera data: {sim.INPUT}')
    rng = random.Random(sim.SEED)
    groups = sim.load_camera()
    world = sim.make_world(groups, rng)
    scenes, _, _ = sim.make_scenes(groups, world, rng)
    if not scenes:
        raise ValueError('No scenes generated')

    results = {}
    for name in ('baseline', 'stable'):
        if name == 'baseline':
            counts, snapshots, events = baseline.run(scenes, False)
        else:
            counts, snapshots, events = run_stable(scenes)
        results[name] = counts
        write_csv(OUT / f'{name}_snapshots.csv', snapshots,
                  ['frame','time_s','fused_id','camera_track_id','status','source',
                   'range_m','radial_velocity_mps','camera_age_ms','radar_age_ms',
                   'measurement_is_synthetic','warning_authorized'])
        write_csv(OUT / f'{name}_events.csv', events,
                  ['frame','time_s','event','track_id','other_track_id','camera_id',
                   'radar_id','truth_debug','donor_truth_history','receiver_truth_history'])

    fields = ['method','created','expired','radar_to_camera_adoptions','merged',
              'ambiguous_adoptions_rejected','fused_updates','camera_only_updates',
              'radar_only_updates','correct_pairs','incorrect_pairs','time_rejected',
              'fragmentation_extra_ids','mixed_truth_tracks','duplicate_truth_frames',
              'confirmed_clutter_active_at_end']
    write_csv(OUT / 'comparison.csv',
              [dict(method=name, **{key: counts[key] for key in fields[1:]})
               for name, counts in results.items()], fields)

    print('\n========== STABLE TRACK MANAGER ==========')
    print(f'Simulated frames: {len(scenes)}')
    for name, c in results.items():
        print(f'{name:9} | Created={c["created"]:4} | Expired={c["expired"]:4} '
              f'| Adopted={c["radar_to_camera_adoptions"]:3} | Merged={c["merged"]:3}')
        print(f'          | Extra truth IDs={c["fragmentation_extra_ids"]:4} '
              f'| Mixed-truth tracks={c["mixed_truth_tracks"]:3} '
              f'| Duplicate truth-frames={c["duplicate_truth_frames"]:4}')
        print(f'          | Correct pairs={c["correct_pairs"]:4} '
              f'| Incorrect pairs={c["incorrect_pairs"]:4}')
    print(f'Saved: {OUT}')
    print('Synthetic truth is used for evaluation only; warnings disabled.')
    print('NOTE: This version avoids post-hoc merges; adoption is conservative.')

if __name__ == '__main__':
    main()
