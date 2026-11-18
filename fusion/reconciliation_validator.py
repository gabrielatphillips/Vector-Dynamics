"""Vector Dynamics: inspect existing reconciliation experiment CSVs.

Read-only diagnostics. Synthetic experiment; no safety authorization.
Run: python fusion/reconciliation_validator.py
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import math

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'fusion' / 'reconciliation_results'
OUT = ROOT / 'fusion' / 'reconciliation_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run track_reconciliation.py first.')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def write(name, rows, fields):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def analyze(name):
    snapshots = load(f'{name}_snapshots.csv')
    events = load(f'{name}_events.csv')
    by_id = defaultdict(list)
    by_frame = defaultdict(list)
    camera_owners = defaultdict(set)
    for s in snapshots:
        by_id[s['fused_id']].append(s)
        by_frame[s['frame']].append(s)
        cid = s.get('camera_track_id', '').strip()
        if cid:
            camera_owners[cid].add(s['fused_id'])
    for records in by_id.values():
        records.sort(key=lambda x: (float(x['time_s']), int(x['frame'])))

    event_counts = Counter(e['event'] for e in events)
    # A camera track assigned to multiple fused IDs over time is a possible
    # identity continuity issue, but is not proof of an incorrect merge.
    camera_changes = [
        {'method': name, 'camera_track_id': cid,
         'persistent_id_count': len(ids), 'persistent_ids': ';'.join(sorted(ids, key=int))}
        for cid, ids in camera_owners.items() if len(ids) > 1
    ]
    camera_changes.sort(key=lambda r: -r['persistent_id_count'])

    merges = []
    for e in events:
        if e['event'] not in ('merged', 'radar_to_camera_adoption'):
            continue
        receiver = e['track_id']
        donor = e.get('other_track_id', '').strip()
        frame = int(e['frame'])
        t = float(e['time_s'])
        receiver_rows = [r for r in by_id.get(receiver, []) if int(r['frame']) == frame]
        receiver_row = receiver_rows[-1] if receiver_rows else None
        # Last donor snapshot strictly before merge; donor is removed at merge.
        donor_prior = [r for r in by_id.get(donor, []) if float(r['time_s']) < t] if donor else []
        donor_row = donor_prior[-1] if donor_prior else None
        age = t - float(donor_row['time_s']) if donor_row else None
        rd = number(receiver_row.get('range_m')) if receiver_row else None
        dd = number(donor_row.get('range_m')) if donor_row else None
        rv = number(receiver_row.get('radial_velocity_mps')) if receiver_row else None
        dv = number(donor_row.get('radial_velocity_mps')) if donor_row else None
        flags = []
        if donor and donor_row is None:
            flags.append('no_prior_donor_snapshot')
        if age is not None and age > 0.2:
            flags.append('donor_snapshot_old')
        if rd is not None and dd is not None and abs(rd - dd) > 5:
            flags.append('range_disagreement_over_5m')
        if rv is not None and dv is not None and abs(rv - dv) > 3:
            flags.append('speed_disagreement_over_3mps')
        merges.append({
            'method': name, 'frame': frame, 'time_s': t,
            'event': e['event'], 'receiver_id': receiver,
            'donor_id': donor, 'camera_id': e.get('camera_id', ''),
            'radar_id': e.get('radar_id', ''),
            'truth_debug_at_event': e.get('truth_debug', ''),
            'donor_snapshot_age_ms': round(age * 1000, 2) if age is not None else '',
            'receiver_range_m': rd if rd is not None else '',
            'donor_prior_range_m': dd if dd is not None else '',
            'range_difference_m': round(abs(rd-dd), 3) if rd is not None and dd is not None else '',
            'speed_difference_mps': round(abs(rv-dv), 3) if rv is not None and dv is not None else '',
            'review_flags': ';'.join(flags)
        })

    # Detect simultaneous duplicate ownership of one camera ID (observable
    # in saved snapshots). This does not use simulator ground truth.
    duplicate_camera_frames = []
    for frame, rows in by_frame.items():
        owners = defaultdict(set)
        for r in rows:
            cid = r.get('camera_track_id', '').strip()
            if cid:
                owners[cid].add(r['fused_id'])
        for cid, ids in owners.items():
            if len(ids) > 1:
                duplicate_camera_frames.append({
                    'method': name, 'frame': frame, 'camera_track_id': cid,
                    'fused_id_count': len(ids), 'fused_ids': ';'.join(sorted(ids, key=int))
                })

    track_rows = []
    for tid, rows in by_id.items():
        first = float(rows[0]['time_s'])
        last = float(rows[-1]['time_s'])
        sources = Counter(r['source'] for r in rows)
        statuses = Counter(r['status'] for r in rows)
        track_rows.append({
            'method': name, 'fused_id': tid, 'first_s': first,
            'last_s': last, 'duration_s': round(last-first, 3),
            'snapshot_count': len(rows),
            'camera_ids_seen': ';'.join(sorted({r['camera_track_id'] for r in rows if r['camera_track_id']}, key=int)),
            'confirmed_snapshots': statuses['confirmed'],
            'tentative_snapshots': statuses['tentative'],
            'coasting_snapshots': statuses['coasting'],
            'fused_snapshots': sources['fused'],
            'radar_only_snapshots': sources['radar_only'],
            'camera_only_snapshots': sources['camera_only'],
            'no_fresh_sensor_snapshots': sources['none']
        })
    track_rows.sort(key=lambda r: (r['snapshot_count'], r['fused_id']))

    short_lived = sum(r['duration_s'] < 0.2 for r in track_rows)
    summary = {
        'method': name,
        'snapshot_rows': len(snapshots),
        'persistent_ids_in_snapshots': len(by_id),
        'event_rows': len(events),
        'merges': event_counts['merged'],
        'adoptions': event_counts['radar_to_camera_adoption'],
        'expired_events': event_counts['expired'],
        'camera_ids_with_multiple_persistent_ids': len(camera_changes),
        'simultaneous_duplicate_camera_frames': len(duplicate_camera_frames),
        'short_lived_tracks_under_200ms': short_lived,
        'merge_events_flagged_for_review': sum(bool(m['review_flags']) for m in merges)
    }
    return summary, merges, camera_changes, duplicate_camera_frames, track_rows


def main():
    all_summaries, all_merges, all_camera, all_dups, all_tracks = [], [], [], [], []
    for method in ('baseline', 'reconciled'):
        summary, merges, camera, duplicates, tracks = analyze(method)
        all_summaries.append(summary)
        all_merges.extend(merges)
        all_camera.extend(camera)
        all_dups.extend(duplicates)
        all_tracks.extend(tracks)

    write('summary.csv', all_summaries, list(all_summaries[0]))
    write('merge_review.csv', all_merges, [
        'method','frame','time_s','event','receiver_id','donor_id','camera_id',
        'radar_id','truth_debug_at_event','donor_snapshot_age_ms',
        'receiver_range_m','donor_prior_range_m','range_difference_m',
        'speed_difference_mps','review_flags'])
    write('camera_id_fragmentation.csv', all_camera,
          ['method','camera_track_id','persistent_id_count','persistent_ids'])
    write('duplicate_camera_frames.csv', all_dups,
          ['method','frame','camera_track_id','fused_id_count','fused_ids'])
    write('track_lifetimes.csv', all_tracks, [
        'method','fused_id','first_s','last_s','duration_s','snapshot_count',
        'camera_ids_seen','confirmed_snapshots','tentative_snapshots',
        'coasting_snapshots','fused_snapshots','radar_only_snapshots',
        'camera_only_snapshots','no_fresh_sensor_snapshots'])

    print('\n========== RECONCILIATION VALIDATOR ==========')
    for s in all_summaries:
        print(f"{s['method']:11} | IDs={s['persistent_ids_in_snapshots']:4} | "
              f"Merges={s['merges']:3} | Adoptions={s['adoptions']:2} | "
              f"Short-lived={s['short_lived_tracks_under_200ms']:3}")
        print(f"             | Camera IDs reused across fused IDs="
              f"{s['camera_ids_with_multiple_persistent_ids']} | "
              f"Simultaneous camera duplicates={s['simultaneous_duplicate_camera_frames']}")
        print(f"             | Merge/adoption events flagged for review="
              f"{s['merge_events_flagged_for_review']}")
    print(f'Files saved: {OUT}')
    print('LIMITATION: Event truth_debug is the current radar measurement truth,')
    print('not donor/receiver identity history. Incorrect merges cannot be')
    print('reliably labeled from these CSVs alone.')
    print('Synthetic data only; no safety warnings authorized.')


if __name__ == '__main__':
    main()
