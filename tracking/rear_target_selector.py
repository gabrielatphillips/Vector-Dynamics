"""Vector Dynamics rear target selection (simulation / algorithm prototype).

Consumes tracks with calibrated lateral position in the truck's lane-relative
coordinate system. This module does NOT infer lane geometry from raw radar
bearing and is NOT suitable for safety decisions without validation.

Run: python tracking/rear_target_selector.py --self-test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class SelectionStatus(str, Enum):
    TRACKED = "tracked"
    COASTING = "coasting"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RearTrack:
    track_id: int
    range_m: float
    closing_mps: float
    lateral_m: float  # signed displacement from estimated ego-lane center
    timestamp_ms: int
    confirmed: bool = True
    lane_geometry_valid: bool = True
    lateral_uncertainty_m: float = 0.0  # nonnegative conservative bound
    range_uncertainty_m: float = 0.0


@dataclass(frozen=True)
class Selection:
    status: SelectionStatus
    timestamp_ms: int
    track_id: Optional[int] = None
    range_m: Optional[float] = None
    closing_mps: Optional[float] = None
    reason: str = ""


class RearTargetSelector:
    """Select the nearest unambiguously in-lane, confirmed, fresh target.

    Uses an explicit lane half-width; geometry must be provided by a separate
    calibrated lane-estimation subsystem. Ambiguous lane overlap returns
    UNKNOWN, rather than treating the ambiguous target as absent.
    """

    def __init__(self, lane_half_width_m: float = 1.75,
                 max_range_m: float = 60.96, min_range_m: float = 4.572,
                 max_age_ms: int = 300, coast_ms: int = 250,
                 switch_margin_m: float = 2.0):
        if not (0 < lane_half_width_m and 0 < min_range_m < max_range_m):
            raise ValueError("Invalid lane width or range")
        if not (0 < max_age_ms and 0 <= coast_ms <= max_age_ms and switch_margin_m >= 0):
            raise ValueError("Invalid timing or switch margin")
        self.half_width = lane_half_width_m
        self.min_range = min_range_m
        self.max_range = max_range_m
        self.max_age = max_age_ms
        self.coast_ms = coast_ms
        self.switch_margin = switch_margin_m
        self.selected_id: Optional[int] = None
        self.last_seen_ms: Optional[int] = None

    def _clear(self, now_ms: int, reason: str) -> Selection:
        self.selected_id = None
        self.last_seen_ms = None
        return Selection(SelectionStatus.UNKNOWN, now_ms, reason=reason)

    def select(self, tracks: Iterable[RearTrack], now_ms: int,
               lane_geometry_valid: bool = True) -> Selection:
        if not lane_geometry_valid:
            return self._clear(now_ms, "lane geometry unavailable")
        eligible = []
        ambiguous = []
        seen_ids = set()
        for t in tracks:
            if t.track_id in seen_ids:
                return self._clear(now_ms, "duplicate track IDs")
            seen_ids.add(t.track_id)
            age = now_ms - t.timestamp_ms
            if age < 0 or age > self.max_age or not t.confirmed:
                continue
            if not t.lane_geometry_valid or t.lateral_uncertainty_m < 0 or t.range_uncertainty_m < 0:
                return self._clear(now_ms, "invalid track geometry")
            if not all(map(__import__('math').isfinite, (t.range_m, t.closing_mps,
                       t.lateral_m, t.lateral_uncertainty_m, t.range_uncertainty_m))):
                return self._clear(now_ms, "nonfinite measurement")
            # Reject only targets definitely outside the operational range.
            if t.range_m + t.range_uncertainty_m < self.min_range or t.range_m - t.range_uncertainty_m > self.max_range:
                continue
            outer = abs(t.lateral_m) + t.lateral_uncertainty_m
            inner = max(0.0, abs(t.lateral_m) - t.lateral_uncertainty_m)
            if outer < self.half_width and (t.range_m - t.range_uncertainty_m >= self.min_range) and (t.range_m + t.range_uncertainty_m <= self.max_range):
                eligible.append(t)
            elif inner <= self.half_width:
                ambiguous.append(t)

        # Any ambiguous track that could be closer than selected candidate
        # prevents confident identification of the nearest following vehicle.
        eligible.sort(key=lambda t: t.range_m)
        best = eligible[0] if eligible else None
        if ambiguous and (best is None or any(t.range_m - t.range_uncertainty_m <= best.range_m + best.range_uncertainty_m for t in ambiguous)):
            return self._clear(now_ms, "closer possible in-lane target is ambiguous")
        if best is None:
            return self._clear(now_ms, "no confirmed in-lane target")

        current = next((t for t in eligible if t.track_id == self.selected_id), None)
        if current is not None and current.track_id != best.track_id:
            # Keep current identity if candidates are near-equal in range.
            if current.range_m - best.range_m <= self.switch_margin:
                best = current
        age = now_ms - best.timestamp_ms
        if age > self.coast_ms:
            return self._clear(now_ms, "measurement too old for display")
        self.selected_id = best.track_id
        self.last_seen_ms = best.timestamp_ms
        if age > 0:
            # Do not show stale distance/speed as if freshly measured.
            return Selection(SelectionStatus.COASTING, now_ms, best.track_id,
                             reason=f"track retained, measurement age {age} ms")
        return Selection(SelectionStatus.TRACKED, now_ms, best.track_id,
                         best.range_m, best.closing_mps, "confirmed in-lane target")


def self_test() -> None:
    def t(i, r, x, now=1000, **kw):
        return RearTrack(i, r, 3.0, x, now, **kw)
    s = RearTargetSelector()
    assert s.select([t(1, 20, 0)], 1000).track_id == 1
    assert s.select([t(2, 10, 3.5), t(1, 20, 0)], 1000).track_id == 1
    assert s.select([t(3, 35, 0), t(1, 20, 0)], 1000).track_id == 1
    assert s.select([t(1, 20, 3.5)], 1000).status == SelectionStatus.UNKNOWN
    assert s.select([t(4, 12, 1.7, lateral_uncertainty_m=.3), t(5, 20, 0)], 1000).status == SelectionStatus.UNKNOWN
    assert s.select([t(5, 20, 0)], 1000, lane_geometry_valid=False).status == SelectionStatus.UNKNOWN
    assert s.select([t(5, 20, 0, now=850)], 1000).status == SelectionStatus.COASTING
    assert s.select([t(5, 20, 0, now=850)], 1000).range_m is None
    assert s.select([t(5, 20, 0, now=600)], 1000).status == SelectionStatus.UNKNOWN
    assert s.select([t(5, 20, 0, confirmed=False)], 1000).status == SelectionStatus.UNKNOWN
    assert s.select([t(6, 20, 0)], 1000).track_id == 6
    assert s.select([t(6, 20, 0), t(7, 19, 0)], 1000).track_id == 6
    assert s.select([t(6, 20, 0), t(7, 15, 0)], 1000).track_id == 7
    assert s.select([t(7, 15, 0), t(7, 18, 0)], 1000).status == SelectionStatus.UNKNOWN
    print("Rear target selector: 13 scenario checks PASS")
    print("Simulation only; no safety warnings authorized")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print('Import RearTargetSelector and RearTrack; or run --self-test')
