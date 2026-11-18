"""Dependency-free integration acceptance checks; run from project root."""
from __future__ import annotations
import csv
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for name in ('communication','fusion','tracking','safety'):
    sys.path.insert(0,str(ROOT/name))
from vehicle_protocol import test_protocol, decode, RANGE_VALID, VELOCITY_VALID, Module, Receiver
from rear_oriented_fixture import build
from rear_cab_integration import read_rear
from cab_state_monitor import CabMonitor, State

def main():
    count=0
    test_protocol();count+=1
    with tempfile.TemporaryDirectory() as tmp:
        rows=build(Path(tmp));count+=1
        assert len(rows)==150 and sum(int(x['valid']) for x in rows)==80;count+=1
        packet_rows=read_rear(Path(tmp)/'rear_selection_timeline.csv')
        assert len(packet_rows)==150;count+=1
        states={state for _,state in packet_rows.values()}
        assert {'TRACKED','LANE UNKNOWN','NO TARGET'}.issubset(states),states;count+=1
        for raw,_ in packet_rows.values():
            pkt=decode(raw)
            assert pkt.source==Module.REAR
            assert len(raw)==32
            if pkt.flags & RANGE_VALID: assert pkt.flags & VELOCITY_VALID
        count+=1
        rx=Receiver(max_age_ms=300)
        raw=next(raw for raw,state in packet_rows.values() if state=='TRACKED')
        rx.ingest(raw,1000)
        assert rx.current(1299) and not rx.current(1301);count+=1
        corrupt=bytearray(raw);corrupt[5]^=1
        try: decode(bytes(corrupt))
        except ValueError: count+=1
        else: raise AssertionError('Corrupt packet accepted')
        monitor=CabMonitor()
        assert monitor.update(1000,packet_valid=True,lane_known=True,target_present=True,track_id=1,range_m=20,closing_mps=2).state==State.TRACKED;count+=1
        assert monitor.current(1301).state==State.SIGNAL_LOST;count+=1
        assert monitor.update(1400,packet_valid=True,lane_known=False,target_present=True).state==State.LANE_UNKNOWN;count+=1
        assert monitor.update(1500,packet_valid=True,lane_known=True,target_present=False).state==State.NO_TARGET;count+=1
        assert monitor.update(1600,packet_valid=True,lane_known=True,target_present=True,track_id=1,range_m=100,closing_mps=2).state==State.LANE_UNKNOWN;count+=1
        assert not monitor.current(1600).warning_authorized and not monitor.current(1600).passing_authorized;count+=1
    print(f'SOFTWARE ACCEPTANCE: {count} checks PASS')
    print('Simulation and desktop only. Physical hardware and field validation remain.')
if __name__=='__main__':main()
