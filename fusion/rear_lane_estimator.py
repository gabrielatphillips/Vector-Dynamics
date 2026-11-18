"""Vector Dynamics rear-lane estimator (simulation and geometry interface).

Converts rear radar range/bearing into lateral position and compares it with
an externally supplied, distance-dependent lane center and uncertainty.

No camera lane detector is implemented here. No safety warnings authorized.
Run: python fusion/rear_lane_estimator.py --self-test
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import argparse
import math


class LaneMembership(str, Enum):
    IN_LANE = "in_lane"
    OUT_OF_LANE = "out_of_lane"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RadarTarget:
    track_id: int
    range_m: float
    bearing_deg: float  # 0 = directly behind; positive = sensor's right
    timestamp_ms: int
    bearing_std_deg: float = 1.0
    range_std_m: float = 0.5


@dataclass(frozen=True)
class LaneEstimate:
    # At target longitudinal distance: lane center relative to radar, meters.
    center_offset_m: float
    lane_width_m: float
    center_std_m: float
    timestamp_ms: int
    confidence: float  # [0,1], externally supplied, NOT calculated from video


@dataclass(frozen=True)
class LaneResult:
    track_id: int
    membership: LaneMembership
    lateral_offset_m: float | None
    relative_to_center_m: float | None
    uncertainty_m: float | None
    confidence: float
    reason: str


def estimate_lane_membership(
    target: RadarTarget,
    lane: LaneEstimate | None,
    now_ms: int,
    *,
    max_radar_age_ms: int = 300,
    max_lane_age_ms: int = 300,
    min_lane_confidence: float = 0.75,
    vehicle_half_width_m: float = 0.95,
    guard_sigma: float = 2.0,
) -> LaneResult:
    """Conservatively classify target footprint relative to lane boundaries.

    Lateral coordinate x = range*sin(bearing). A complete vehicle footprint,
    expanded by uncertainty, must fit inside lane to qualify IN_LANE.
    OUT_OF_LANE requires the expanded footprint entirely outside the lane.
    All overlap/boundary cases are UNKNOWN.

    A supplied lane center must correspond to this target's longitudinal range.
    This module does not infer curved-road lane geometry from radar alone.
    """
    def unknown(reason: str, x=None, relative=None, uncertainty=None):
        return LaneResult(target.track_id, LaneMembership.UNKNOWN, x, relative,
                          uncertainty, 0.0, reason)

    if now_ms < target.timestamp_ms or now_ms - target.timestamp_ms > max_radar_age_ms:
        return unknown("radar_stale_or_future")
    if not all(math.isfinite(v) for v in (target.range_m, target.bearing_deg,
                                         target.bearing_std_deg, target.range_std_m)):
        return unknown("invalid_radar")
    if target.range_m <= 0 or target.range_std_m < 0 or target.bearing_std_deg < 0:
        return unknown("invalid_radar")
    if lane is None:
        return unknown("lane_geometry_missing")
    if now_ms < lane.timestamp_ms or now_ms - lane.timestamp_ms > max_lane_age_ms:
        return unknown("lane_geometry_stale_or_future")
    if not all(math.isfinite(v) for v in (lane.center_offset_m, lane.lane_width_m,
                                         lane.center_std_m, lane.confidence)):
        return unknown("invalid_lane_geometry")
    if lane.lane_width_m <= 0 or lane.center_std_m < 0 or not 0 <= lane.confidence <= 1:
        return unknown("invalid_lane_geometry")
    if lane.confidence < min_lane_confidence:
        return unknown("lane_confidence_low")
    if vehicle_half_width_m <= 0 or guard_sigma < 0:
        raise ValueError("vehicle_half_width_m must be positive and guard_sigma nonnegative")

    theta = math.radians(target.bearing_deg)
    x = target.range_m * math.sin(theta)
    relative = x - lane.center_offset_m
    # First-order uncertainty propagation. Angle uncertainty grows with range.
    sigma = math.sqrt((math.sin(theta)*target.range_std_m)**2 +
                      (target.range_m*math.cos(theta)*math.radians(target.bearing_std_deg))**2 +
                      lane.center_std_m**2)
    expanded = vehicle_half_width_m + guard_sigma * sigma
    half_lane = lane.lane_width_m / 2
    if abs(relative) + expanded < half_lane:
        membership, reason = LaneMembership.IN_LANE, "footprint_inside_lane"
    elif abs(relative) - expanded > half_lane:
        membership, reason = LaneMembership.OUT_OF_LANE, "footprint_outside_lane"
    else:
        membership, reason = LaneMembership.UNKNOWN, "lane_boundary_ambiguous"
    return LaneResult(target.track_id, membership, x, relative, sigma,
                      lane.confidence, reason)


def self_test():
    lane = LaneEstimate(0, 3.7, .10, 1000, .98)
    def t(id, distance, angle, stamp=1000, std=.2):
        return RadarTarget(id, distance, angle, stamp, std, .2)
    cases = [
        ("center following", t(1, 25, 0), lane, 1100, LaneMembership.IN_LANE),
        ("adjacent right", t(2, 25, 9), lane, 1100, LaneMembership.OUT_OF_LANE),
        ("adjacent left", t(3, 25, -9), lane, 1100, LaneMembership.OUT_OF_LANE),
        ("lane edge ambiguous", t(4, 25, 4), lane, 1100, LaneMembership.UNKNOWN),
        ("missing lane", t(5, 25, 0), None, 1100, LaneMembership.UNKNOWN),
        ("stale lane", t(6, 25, 0), lane, 1400, LaneMembership.UNKNOWN),
        ("stale radar", t(7, 25, 0, 500), lane, 1100, LaneMembership.UNKNOWN),
        ("low confidence", t(8, 25, 0), LaneEstimate(0,3.7,.1,1000,.4), 1100, LaneMembership.UNKNOWN),
        ("curve corrected", t(9, 25, 7), LaneEstimate(3.05,3.7,.1,1000,.98), 1100, LaneMembership.IN_LANE),
        ("curve unmodeled", t(10, 25, 7), lane, 1100, LaneMembership.UNKNOWN),
        ("far uncertain", t(11, 60, 0, std=2), lane, 1100, LaneMembership.UNKNOWN),
        ("invalid range", t(12, -1, 0), lane, 1100, LaneMembership.UNKNOWN),
        ("future timestamp", t(13, 25, 0, 1200), lane, 1100, LaneMembership.UNKNOWN),
    ]
    for name, target, geometry, now, expected in cases:
        actual = estimate_lane_membership(target, geometry, now)
        assert actual.membership == expected, f"{name}: {actual.membership} != {expected} ({actual.reason})"
    print(f"Rear lane estimator: {len(cases)} scenario checks PASS")
    print("Lane center inputs are synthetic; no real camera calibration or safety warnings authorized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print("Use --self-test to validate synthetic lane classification scenarios")
