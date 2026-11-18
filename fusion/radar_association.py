from pathlib import Path
from collections import defaultdict
import csv
import math
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'vision' / 'results' / 'tracking_data.csv'
OUT = ROOT / 'fusion' / 'phase12_results'
SEED = 2026
IMAGE_WIDTH = 1920
HFOV_DEG = 70.0
MAX_BEARING_DIFF = 5.0
RADAR_DETECTION_PROB = 0.90
CAMERA_DETECTION_PROB = 0.88
CLUTTER_PROB = 0.20
RADAR_ONLY_COUNT = 4
RADAR_TIME_OFFSET_S = 0.045
ANGLE_STD_DEG = 1.0
RANGE_STD_M = 1.0
SPEED_STD_MPS = 0.45
FRONT_MAX_M = 121.92
GATE_COST = 8.0
UNMATCHED_COST = 6.0
HISTORY_TTL_S = 0.8


def bearing(x):
    focal = (IMAGE_WIDTH / 2) / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan((x - IMAGE_WIDTH / 2) / focal))


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_camera():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing input: {INPUT}')
    groups = defaultdict(list)
    with INPUT.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            groups[int(row['frame'])].append({
                'frame': int(row['frame']),
                'time_s': float(row['timestamp_s']),
                'id': int(row['track_id']),
                'bearing': bearing(float(row['center_x'])),
                'confidence': float(row['confidence']),
            })
    return dict(sorted(groups.items()))


def make_world(groups, rng):
    ids = sorted({o['id'] for frame in groups.values() for o in frame})
    first = {}
    for obs in groups.values():
        for o in obs:
            first[o['id']] = min(first.get(o['id'], o['time_s']), o['time_s'])
    world = {}
    for tid in ids:
        world[tid] = {
            'range0': rng.uniform(40, 100),
            'speed': rng.uniform(-4.0, 2.0),
            't0': first[tid],
        }
    # Independent synthetic radar-only objects with their own motion and bearings.
    for i in range(RADAR_ONLY_COUNT):
        tid = -(i + 1)
        world[tid] = {
            'range0': rng.uniform(45, 95),
            'speed': rng.uniform(-2.0, 1.0),
            't0': 0.0,
            'bearing0': rng.uniform(-24, 24),
            'bearing_rate': rng.uniform(-0.12, 0.12),
        }
    return world


def make_scenes(groups, world, rng):
    scenes = []
    all_cam = []
    all_radar = []
    frames = sorted(groups)
    if not frames:
        raise ValueError('No frames in tracking_data.csv')
    for frame in frames:
        original = groups[frame]
        t = original[0]['time_s']
        cams = []
        radar = []
        for obj in original:
            tid = obj['id']
            w = world[tid]
            if rng.random() < CAMERA_DETECTION_PROB:
                cams.append(dict(obj))
            # Radar generation is independent of camera detection sampling,
            # though the shared synthetic target bearing comes from the source video.
            if rng.random() < RADAR_DETECTION_PROB:
                rt = t + RADAR_TIME_OFFSET_S + rng.gauss(0, 0.012)
                true_range = w['range0'] + w['speed'] * (rt - w['t0'])
                if 0 < true_range <= FRONT_MAX_M:
                    radar.append({
                        'frame': frame, 'time_s': rt, 'truth_id': tid,
                        'angle_deg': obj['bearing'] + rng.gauss(0, ANGLE_STD_DEG),
                        'range_m': max(0, true_range + rng.gauss(0, RANGE_STD_M)),
                        'velocity_mps': w['speed'] + rng.gauss(0, SPEED_STD_MPS),
                        'kind': 'camera-origin-world',
                    })
        # Radar-only objects appear at a lower frame rate to mimic a separate radar.
        if frame % 3 == 0:
            for tid in range(-RADAR_ONLY_COUNT, 0):
                w = world[tid]
                if rng.random() < RADAR_DETECTION_PROB:
                    rt = t + RADAR_TIME_OFFSET_S
                    radar.append({
                        'frame': frame, 'time_s': rt, 'truth_id': tid,
                        'angle_deg': w['bearing0'] + w['bearing_rate'] * rt + rng.gauss(0, ANGLE_STD_DEG),
                        'range_m': max(0, w['range0'] + w['speed'] * rt + rng.gauss(0, RANGE_STD_M)),
                        'velocity_mps': w['speed'] + rng.gauss(0, SPEED_STD_MPS),
                        'kind': 'radar-only',
                    })
        if rng.random() < CLUTTER_PROB:
            radar.append({
                'frame': frame, 'time_s': t + RADAR_TIME_OFFSET_S,
                'truth_id': None, 'angle_deg': rng.uniform(-35, 35),
                'range_m': rng.uniform(5, FRONT_MAX_M),
                'velocity_mps': rng.uniform(-10, 10), 'kind': 'clutter',
            })
        for ri, r in enumerate(radar):
            r['radar_id'] = f'{frame}:{ri}'
        scenes.append((frame, t, cams, radar))
        all_cam.extend(cams)
        all_radar.extend(radar)
    return scenes, all_cam, all_radar


def greedy(cams, radar):
    options = []
    for i, c in enumerate(cams):
        for j, r in enumerate(radar):
            diff = abs(c['bearing'] - r['angle_deg'])
            if diff <= MAX_BEARING_DIFF:
                options.append((diff, i, j))
    options.sort()
    used_c, used_r = set(), set()
    matches = []
    for cost, i, j in options:
        if i not in used_c and j not in used_r:
            matches.append((i, j, cost))
            used_c.add(i)
            used_r.add(j)
    return matches


def global_assign(cams, radar, history, t):
    """Exact min-cost bipartite matching via scipy if available,
    otherwise a pure-Python min-cost flow algorithm. Unmatched allowed.
    """
    n, m = len(cams), len(radar)
    if not n or not m:
        return []
    costs = np.full((n, m), np.inf)
    for i, c in enumerate(cams):
        prev = history.get(c['id'])
        for j, r in enumerate(radar):
            angle = abs(c['bearing'] - r['angle_deg'])
            if angle > MAX_BEARING_DIFF:
                continue
            cost = angle
            if prev is not None and 0 <= t - prev['time_s'] <= HISTORY_TTL_S:
                expected = prev['range_m'] + prev['velocity_mps'] * (r['time_s'] - prev['time_s'])
                residual = abs(r['range_m'] - expected)
                # Avoid hard rejection on history: track changes may be abrupt.
                cost += min(residual / 3.0, 10.0)
            if cost < GATE_COST:
                costs[i, j] = cost
    # Min-cost flow with unmatched camera option: source->camera->radar->sink,
    # or source->camera->sink at UNMATCHED_COST. All cameras send one unit.
    source = n + m
    sink = source + 1
    nodes = sink + 1
    graph = [[] for _ in range(nodes)]
    def add(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
    for i in range(n):
        add(source, i, 1, 0.0)
        add(i, sink, 1, UNMATCHED_COST)
    for j in range(m):
        add(n + j, sink, 1, 0.0)
    for i in range(n):
        for j in range(m):
            if np.isfinite(costs[i, j]) and costs[i, j] < UNMATCHED_COST:
                add(i, n + j, 1, float(costs[i, j]))
    for _ in range(n):
        dist = [float('inf')] * nodes
        prev = [None] * nodes
        dist[source] = 0.0
        # Bellman-Ford supports reverse edges with negative costs.
        for _pass in range(nodes - 1):
            changed = False
            for u in range(nodes):
                if not math.isfinite(dist[u]):
                    continue
                for ei, edge in enumerate(graph[u]):
                    v, cap, cost, _ = edge
                    if cap > 0 and dist[u] + cost < dist[v] - 1e-10:
                        dist[v] = dist[u] + cost
                        prev[v] = (u, ei)
                        changed = True
            if not changed:
                break
        if prev[sink] is None:
            raise RuntimeError('Flow assignment unexpectedly failed')
        v = sink
        while v != source:
            u, ei = prev[v]
            edge = graph[u][ei]
            edge[1] -= 1
            graph[v][edge[3]][1] += 1
            v = u
    matches = []
    for i in range(n):
        for edge in graph[i]:
            v, cap, cost, _ = edge
            if n <= v < n + m and cap == 0:
                matches.append((i, v - n, costs[i, v - n]))
    return matches


def evaluate(scenes, method):
    history = {}
    rows = []
    totals = defaultdict(int)
    for frame, t, cams, radar in scenes:
        matches = greedy(cams, radar) if method == 'greedy' else global_assign(cams, radar, history, t)
        used_c = {i for i, _, _ in matches}
        used_r = {j for _, j, _ in matches}
        camera_ids = {c['id'] for c in cams}
        # Eligible true associations are radar measurements with camera counterpart.
        eligible = sum(r['truth_id'] in camera_ids for r in radar if r['truth_id'] is not None)
        totals['eligible'] += eligible
        totals['radar'] += len(radar)
        totals['camera'] += len(cams)
        totals['matches'] += len(matches)
        totals['unmatched_camera'] += len(cams) - len(used_c)
        totals['unmatched_radar'] += len(radar) - len(used_r)
        totals['radar_only'] += sum(r['kind'] == 'radar-only' for r in radar)
        totals['clutter'] += sum(r['kind'] == 'clutter' for r in radar)
        for i, j, cost in matches:
            c, r = cams[i], radar[j]
            correct = r['truth_id'] is not None and c['id'] == r['truth_id']
            totals['correct' if correct else 'incorrect'] += 1
            rows.append({
                'method': method, 'frame': frame,
                'camera_track_id': c['id'], 'radar_id': r['radar_id'],
                'radar_truth_id': r['truth_id'], 'radar_kind': r['kind'],
                'camera_time_s': round(c['time_s'], 4),
                'radar_time_s': round(r['time_s'], 4),
                'age_difference_ms': round(1000 * (r['time_s'] - c['time_s']), 2),
                'range_m': round(r['range_m'], 3),
                'radial_velocity_mps': round(r['velocity_mps'], 3),
                'angle_deg': round(r['angle_deg'], 3),
                'association_cost': round(float(cost), 4),
                'correct_match': correct,
            })
            if method == 'global':
                history[c['id']] = {
                    'range_m': r['range_m'], 'velocity_mps': r['velocity_mps'],
                    'time_s': r['time_s'],
                }
        # History updates use assigned pairs, not hidden truth labels.
        # A wrong association can therefore contaminate later matches.
    precision = totals['correct'] / totals['matches'] if totals['matches'] else 0
    recall = totals['correct'] / totals['eligible'] if totals['eligible'] else 0
    return rows, totals, precision, recall


def main():
    rng = random.Random(SEED)
    groups = load_camera()
    world = make_world(groups, rng)
    scenes, cams, radar = make_scenes(groups, world, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        'method', 'frame', 'camera_track_id', 'radar_id', 'radar_truth_id',
        'radar_kind', 'camera_time_s', 'radar_time_s', 'age_difference_ms',
        'range_m', 'radial_velocity_mps', 'angle_deg', 'association_cost',
        'correct_match',
    ]
    summary = []
    for method in ('greedy', 'global'):
        rows, totals, precision, recall = evaluate(scenes, method)
        save_csv(OUT / f'{method}_matches.csv', rows, fields)
        summary.append({
            'method': method,
            'precision_pct': round(100 * precision, 3),
            'recall_pct': round(100 * recall, 3),
            **totals,
        })
    save_csv(OUT / 'comparison.csv', summary, list(dict.fromkeys(k for row in summary for k in row)))
    print('\n========== PHASE 12: ROBUST ASSOCIATION ==========')
    print(f'Frames: {len(scenes)} | Camera detections: {len(cams)} | Radar detections: {len(radar)}')
    print(f'Radar-only detections: {summary[0]["radar_only"]} | Clutter detections: {summary[0]["clutter"]}')
    for r in summary:
        print(f'{r["method"]:7} | Precision={r["precision_pct"]:6.2f}% | Recall={r["recall_pct"]:6.2f}% | Correct={r["correct"]:4} | Incorrect={r["incorrect"]:3} | Unmatched radar={r["unmatched_radar"]:4}')
    print(f'Saved: {OUT}')

if __name__ == '__main__':
    main()
