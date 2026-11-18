"""Fail-closed desktop cab-state policy for simulation/bench tests only.

Never emits a driving/passing authorization. Valid measurements are display-only.
"""
from dataclasses import dataclass
from enum import Enum

class State(str,Enum):
    TRACKED='TRACKED'
    NO_TARGET='NO TARGET'
    LANE_UNKNOWN='LANE UNKNOWN'
    SIGNAL_LOST='SIGNAL LOST'

@dataclass(frozen=True)
class DisplayState:
    state: State
    track_id: int|None = None
    range_m: float|None = None
    closing_mps: float|None = None
    warning_authorized: bool = False
    passing_authorized: bool = False

class CabMonitor:
    def __init__(self,timeout_ms=300):
        if timeout_ms <= 0: raise ValueError('timeout must be positive')
        self.timeout_ms=timeout_ms
        self.last_ms=None
        self.last=DisplayState(State.SIGNAL_LOST)

    def update(self, now_ms:int, *, packet_valid:bool, lane_known:bool, target_present:bool,
               track_id:int|None=None, range_m:float|None=None, closing_mps:float|None=None):
        if now_ms < 0 or (self.last_ms is not None and now_ms < self.last_ms):
            raise ValueError('Non-monotonic time')
        self.last_ms=now_ms
        if not packet_valid:
            self.last=DisplayState(State.SIGNAL_LOST)
        elif not lane_known:
            self.last=DisplayState(State.LANE_UNKNOWN)
        elif not target_present:
            self.last=DisplayState(State.NO_TARGET)
        elif (track_id is None or range_m is None or closing_mps is None or
              not 4.572 <= range_m <= 60.96):
            self.last=DisplayState(State.LANE_UNKNOWN)
        else:
            self.last=DisplayState(State.TRACKED,track_id,range_m,closing_mps)
        return self.last

    def current(self,now_ms:int):
        if self.last_ms is None or now_ms < self.last_ms or now_ms-self.last_ms > self.timeout_ms:
            return DisplayState(State.SIGNAL_LOST)
        return self.last
