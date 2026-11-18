"""Vector Dynamics rear detection integration (synthetic scenario harness).

Install in fusion/. Imports existing fusion/rear_lane_estimator.py,
tracking/rear_target_selector.py, communication/vehicle_protocol.py.

Only an explicitly confirmed, in-lane, fresh target is transmitted with
range/velocity-valid flags. Unknown/coasting -> invalid measurement packet.
Radar and lane geometry are synthetic. No safety warnings authorized.
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / 'fusion', ROOT / 'tracking', ROOT / 'communication'):
    sys.path.insert(0, str(folder))
from rear_lane_estimator import (RadarTarget, LaneEstimate, LaneMembership,
                                 estimate_lane_membership)
from rear_target_selector import RearTargetSelector, RearTrack, SelectionStatus
from vehicle_protocol import (VehiclePacket, Module, Status, RANGE_VALID,
                              VELOCITY_VALID, BEARING_VALID, RADAR_FRESH,
                              SYNTHETIC, encode, decode, Receiver)

@dataclass(frozen=True)
class Observation:
    track_id: int
    range_m: float
    bearing_deg: float
    closing_mps: float  # positive means closing; protocol radial is negative
    timestamp_ms: int
    confirmed: bool = True
    bearing_std_deg: float = 0.2
    range_std_m: float = 0.2


class RearDetectionPipeline:
    def __init__(self):
        self.selector = RearTargetSelector()
        self.sequence = 0

    def step(self, observations: list[Observation], lane: LaneEstimate | None,
             now_ms: int) -> tuple[VehiclePacket, dict]:
        tracks = []
        classifications = []
        lane_ok = lane is not None and 0 <= now_ms - lane.timestamp_ms <= 300 and lane.confidence >= .75
        for obs in observations:
            radar = RadarTarget(obs.track_id, obs.range_m, obs.bearing_deg,
                                obs.timestamp_ms, obs.bearing_std_deg, obs.range_std_m)
            result = estimate_lane_membership(radar, lane, now_ms)
            classifications.append((obs.track_id, result.membership.value, result.reason))
            if result.membership == LaneMembership.OUT_OF_LANE:
                continue
            if result.membership == LaneMembership.UNKNOWN:
                # Unknown candidates must block a confident nearest-target choice.
                # Represent ambiguous lateral footprint, except if geometry is globally
                # unavailable, in which case selector fails closed via lane_ok.
                if result.relative_to_center_m is None:
                    lane_ok = False
                    continue
                uncertainty = max(0.0, (lane.lane_width_m / 2) - abs(result.relative_to_center_m) + .05)
            else:
                # Selector uses centerpoint lateral bounds, while estimator already
                # checked footprint. Do not double-count vehicle half-width here.
                uncertainty = max(0.0, result.uncertainty_m or 0.0)
            if result.relative_to_center_m is None:
                lane_ok = False
                continue
            tracks.append(RearTrack(obs.track_id, obs.range_m, obs.closing_mps,
                                    result.relative_to_center_m, obs.timestamp_ms,
                                    confirmed=obs.confirmed,
                                    lateral_uncertainty_m=uncertainty,
                                    range_uncertainty_m=obs.range_std_m))
        selection = self.selector.select(tracks, now_ms, lane_geometry_valid=lane_ok)
        chosen = next((o for o in observations if o.track_id == selection.track_id), None)
        valid = selection.status == SelectionStatus.TRACKED and chosen is not None
        flags = SYNTHETIC
        if valid:
            flags |= RANGE_VALID | VELOCITY_VALID | BEARING_VALID | RADAR_FRESH
        # Unknown and coasting packets have no valid measurements; receiver
        # must clear previously displayed rear values on these packets.
        pkt = VehiclePacket(Module.REAR, Module.CAB, self.sequence, now_ms,
                            selection.track_id or 0,
                            round(selection.range_m * 1000) if valid else 0,
                            round(-selection.closing_mps * 1000) if valid else 0,
                            round(chosen.bearing_deg * 100) if valid else 0,
                            flags,
                            Status.CONFIRMED if valid else Status.COASTING)
        self.sequence += 1
        return pkt, {'time_ms': now_ms, 'status': selection.status.value,
                     'selected_id': selection.track_id,
                     'reason': selection.reason, 'classifications': classifications,
                     'range_m': selection.range_m if valid else None,
                     'closing_mps': selection.closing_mps if valid else None}


def scenarios():
    """Deterministic synthetic tests; fresh pipeline instance per scenario."""
    def lane(t, center=0, confidence=.98):
        return LaneEstimate(center, 3.7, .05, t, confidence)
    def o(i, distance, lateral, t, closing=3.0):
        return Observation(i, distance, math.degrees(math.asin(lateral / distance)), closing, t)
    return [
        ('directly_behind', [o(1,20,0,1000)], lane(1000), 1),
        ('closer_adjacent_left', [o(2,10,-3.5,1000),o(1,20,0,1000)], lane(1000), 1),
        ('closer_adjacent_right', [o(2,10,3.5,1000),o(1,20,0,1000)], lane(1000), 1),
        ('two_same_lane', [o(1,30,0,1000),o(3,15,0,1000)], lane(1000), 3),
        ('three_vehicles', [o(2,12,-3.5,1000),o(3,11,3.5,1000),o(1,25,0,1000)], lane(1000), 1),
        ('boundary_ambiguous', [o(2,12,1.7,1000),o(1,20,0,1000)], lane(1000), None),
        ('lane_missing', [o(1,20,0,1000)], None, None),
        ('lane_low_confidence', [o(1,20,0,1000)], lane(1000,confidence=.2), None),
        ('no_vehicles', [], lane(1000), None),
        ('outside_range', [o(1,80,0,1000)], lane(1000), None),
        ('curved_lane', [o(1,25,3.0,1000)], lane(1000,center=3.0), 1),
    ]


def self_test():
    rx = Receiver(max_age_ms=300)
    for name, obs, geometry, expected in scenarios():
        pipeline = RearDetectionPipeline()
        pkt, info = pipeline.step(obs, geometry, 1000)
        decoded = decode(encode(pkt))
        assert decoded == pkt, name
        assert (info['selected_id'] if pkt.flags & RANGE_VALID else None) == expected, (name, info)
        assert bool(pkt.flags & VELOCITY_VALID) == (expected is not None), name
        assert pkt.flags & SYNTHETIC, name
        # Each scenario uses its own receiver sequence space.
        rx = Receiver()
        rx.ingest(encode(pkt), 1050)
        assert bool(rx.current(1050)), name
    # Explicit stale/coasting transition: old range must not survive.
    pipeline = RearDetectionPipeline()
    initial, _ = pipeline.step([Observation(7,20,0,3,1000)],LaneEstimate(0,3.7,.05,1000,.98),1000)
    unknown, _ = pipeline.step([],LaneEstimate(0,3.7,.05,1100,.98),1100)
    assert initial.flags & RANGE_VALID
    assert not unknown.flags & RANGE_VALID
    rx = Receiver()
    rx.ingest(encode(initial), 1030)
    rx.ingest(encode(unknown), 1130)
    latest = rx.last_packet[(Module.REAR, 0)]
    assert not latest.flags & RANGE_VALID
    assert not rx.current(1500)
    # lane-change transition: previously selected vehicle leaves ego lane.
    pipeline = RearDetectionPipeline()
    first, _ = pipeline.step([Observation(1,20,0,3,1000)],LaneEstimate(0,3.7,.05,1000,.98),1000)
    changed, _ = pipeline.step([Observation(1,20,math.degrees(math.asin(3.5/20)),3,1100)],LaneEstimate(0,3.7,.05,1100,.98),1100)
    assert first.flags & RANGE_VALID and not changed.flags & RANGE_VALID
    print(f'Rear detection pipeline: {len(scenarios())+2} integration checks PASS')
    print('Synthetic radar and lane geometry only; NO safety warnings authorized')


def demo():
    pipeline = RearDetectionPipeline()
    output = ROOT / 'fusion' / 'rear_detection_results'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    rx = Receiver(max_age_ms=300)
    for i in range(150):
        ms = i*100
        geom = LaneEstimate(0,3.7,.05,ms,.98) if not 6000 <= ms < 8000 else None
        following = Observation(1, max(8, 40 - ms/1000),0,2.0,ms)
        left = Observation(2,12,-18,3,ms)  # -10 deg bearing, adjacent
        right = Observation(3,11,20,3,ms)
        observations = [following,left,right] if ms < 10000 else [left,right]
        pkt, info = pipeline.step(observations, geom, ms)
        raw = encode(pkt)
        received = rx.ingest(raw, ms+40)
        rows.append({'time_ms':ms, 'selected_id':info['selected_id'] or '',
                     'status':info['status'], 'range_m':info['range_m'] if info['range_m'] is not None else '',
                     'closing_mps':info['closing_mps'] if info['closing_mps'] is not None else '',
                     'valid':int(bool(received.flags & RANGE_VALID)),
                     'packet_hex':raw.hex(), 'reason':info['reason']})
    with (output/'rear_selection_timeline.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} rear telemetry records: {output / "rear_selection_timeline.csv"}')
    print('Synthetic scenario, not yet connected to the full video fusion manager or cab UI.')
    print('NO safety warnings authorized')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--demo',action='store_true')
    args=parser.parse_args()
    if args.self_test or not args.demo: self_test()
    if args.demo: demo()
