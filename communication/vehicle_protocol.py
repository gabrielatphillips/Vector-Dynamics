"""Vector Dynamics: versioned binary vehicle telemetry protocol (desktop prototype).

No warning authorization. Timestamps are sender monotonic milliseconds; cross-device
age validation requires clock synchronization or receiver-side arrival age.
"""
from __future__ import annotations

import argparse
import random
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"VD"
VERSION = 1
# magic, version, source, destination, sequence, sender_ms, track_id,
# range_mm, radial_mmps, bearing_centideg, flags, status, reserved, crc32
FORMAT_NO_CRC = "<2sBBBIIHiihBBB"
FORMAT = FORMAT_NO_CRC + "I"
PAYLOAD_SIZE = struct.calcsize(FORMAT_NO_CRC)
PACKET_SIZE = struct.calcsize(FORMAT)

class Module(IntEnum):
    FRONT = 1
    REAR = 2
    CAB = 3

class Status(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    COASTING = 2
    EXPIRED = 3

RANGE_VALID = 1 << 0
VELOCITY_VALID = 1 << 1
BEARING_VALID = 1 << 2
CAMERA_FRESH = 1 << 3
RADAR_FRESH = 1 << 4
SYNTHETIC = 1 << 5
KNOWN_FLAGS = RANGE_VALID | VELOCITY_VALID | BEARING_VALID | CAMERA_FRESH | RADAR_FRESH | SYNTHETIC

@dataclass(frozen=True)
class VehiclePacket:
    source: Module
    destination: Module
    sequence: int
    sender_ms: int
    track_id: int
    range_mm: int
    radial_mmps: int  # negative means closing
    bearing_centideg: int
    flags: int
    status: Status

    @property
    def range_m(self):
        return self.range_mm / 1000 if self.flags & RANGE_VALID else None

    @property
    def radial_mps(self):
        return self.radial_mmps / 1000 if self.flags & VELOCITY_VALID else None


def crc32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xffffffff


def validate(packet: VehiclePacket):
    if not isinstance(packet.source, Module) or not isinstance(packet.destination, Module):
        raise ValueError("Invalid module")
    if not isinstance(packet.status, Status):
        raise ValueError("Invalid status")
    for name, value, low, high in (
        ("sequence", packet.sequence, 0, 0xffffffff),
        ("sender_ms", packet.sender_ms, 0, 0xffffffff),
        ("track_id", packet.track_id, 0, 65535),
        ("range_mm", packet.range_mm, -2147483648, 2147483647),
        ("radial_mmps", packet.radial_mmps, -2147483648, 2147483647),
        ("bearing_centideg", packet.bearing_centideg, -32768, 32767),
        ("flags", packet.flags, 0, 255),
    ):
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} outside supported range")
    if packet.flags & ~KNOWN_FLAGS:
        raise ValueError("Unknown flags")
    if packet.flags & RANGE_VALID and packet.range_mm < 0:
        raise ValueError("Valid range cannot be negative")
    if packet.flags & BEARING_VALID and not -18000 <= packet.bearing_centideg <= 18000:
        raise ValueError("Bearing must be within +/-180 degrees")


def encode(packet: VehiclePacket) -> bytes:
    validate(packet)
    data = struct.pack(
        FORMAT_NO_CRC, MAGIC, VERSION, int(packet.source), int(packet.destination),
        packet.sequence, packet.sender_ms, packet.track_id, packet.range_mm,
        packet.radial_mmps, packet.bearing_centideg, packet.flags,
        int(packet.status), 0,
    )
    return data + struct.pack("<I", crc32(data))


def decode(data: bytes) -> VehiclePacket:
    if len(data) != PACKET_SIZE:
        raise ValueError(f"Wrong packet length: {len(data)} != {PACKET_SIZE}")
    if crc32(data[:-4]) != struct.unpack("<I", data[-4:])[0]:
        raise ValueError("CRC mismatch")
    magic, version, source, destination, seq, sender_ms, track_id, distance, speed, bearing, flags, status, reserved, _ = struct.unpack(FORMAT, data)
    if magic != MAGIC or version != VERSION or reserved != 0:
        raise ValueError("Invalid magic/version/reserved")
    try:
        result = VehiclePacket(Module(source), Module(destination), seq, sender_ms,
                               track_id, distance, speed, bearing, flags, Status(status))
    except ValueError as exc:
        raise ValueError("Invalid enum field") from exc
    validate(result)
    return result


class Receiver:
    """Receiver freshness is based on local arrival time, NOT remote sender clock.

    Sequence numbers are checked per source module. A restart needs a session reset
    (or a future protocol revision with a boot/session identifier).
    """
    def __init__(self, max_age_ms=300):
        self.max_age_ms = max_age_ms
        self.last_seq = {}
        self.last_arrival = {}
        self.last_packet = {}
        self.accepted = 0
        self.rejected_replay = 0

    def ingest(self, data: bytes, arrival_ms: int):
        pkt = decode(data)
        last = self.last_seq.get(pkt.source)
        if last is not None:
            delta = (pkt.sequence - last) & 0xffffffff
            if delta == 0 or delta >= 0x80000000:
                self.rejected_replay += 1
                raise ValueError("Duplicate or out-of-order sequence")
        self.last_seq[pkt.source] = pkt.sequence
        key = (pkt.source, pkt.track_id)
        self.last_arrival[key] = arrival_ms
        self.last_packet[key] = pkt
        self.accepted += 1
        return pkt

    def current(self, now_ms: int):
        return {key: pkt for key, pkt in self.last_packet.items()
                if 0 <= now_ms - self.last_arrival[key] <= self.max_age_ms
                and pkt.status != Status.EXPIRED}


def test_protocol():
    pkt = VehiclePacket(Module.FRONT, Module.CAB, 123, 1000, 42, 60000, -4500, 1200,
                        RANGE_VALID | VELOCITY_VALID | BEARING_VALID | RADAR_FRESH | SYNTHETIC,
                        Status.CONFIRMED)
    raw = encode(pkt)
    assert decode(raw) == pkt
    assert len(raw) == PACKET_SIZE
    assert pkt.range_m == 60.0 and pkt.radial_mps == -4.5
    try:
        decode(raw[:-1])
        raise AssertionError("Truncated packet accepted")
    except ValueError:
        pass
    damaged = bytearray(raw)
    damaged[10] ^= 0x40
    try:
        decode(bytes(damaged))
        raise AssertionError("Corrupted packet accepted")
    except ValueError:
        pass
    receiver = Receiver(max_age_ms=300)
    receiver.ingest(raw, arrival_ms=5000)
    assert receiver.current(5299)
    assert not receiver.current(5301)
    try:
        receiver.ingest(raw, arrival_ms=5100)
        raise AssertionError("Replay accepted")
    except ValueError:
        pass
    return pkt


def simulate(count=300, drop_rate=0.10, corruption_rate=0.03, seed=42):
    rng = random.Random(seed)
    receiver = Receiver(max_age_ms=300)
    sent = dropped = corrupted = received = stale_steps = 0
    # Simulate 20Hz transmission, fixed local receive time after 40-120ms delay.
    events = []
    for i in range(count):
        pkt = VehiclePacket(Module.REAR, Module.CAB, i, i * 50, 7, 25000 - min(i, 100) * 80,
                            -1600, -250, RANGE_VALID | VELOCITY_VALID | BEARING_VALID | RADAR_FRESH | SYNTHETIC,
                            Status.CONFIRMED)
        sent += 1
        if rng.random() < drop_rate:
            dropped += 1
            continue
        raw = bytearray(encode(pkt))
        if rng.random() < corruption_rate:
            raw[rng.randrange(PAYLOAD_SIZE)] ^= 1 << rng.randrange(8)
            corrupted += 1
        delivery_ms = i * 50 + rng.randint(40, 120)
        events.append((delivery_ms, bytes(raw)))
    events.sort(key=lambda e: e[0])
    for arrival_ms, data in events:
        try:
            receiver.ingest(data, arrival_ms)
            received += 1
        except ValueError:
            pass
    last_time = count * 50 + 500
    assert not receiver.current(last_time), "Stale data should not remain valid"
    print("========== VEHICLE PROTOCOL ==========")
    print(f"Packet size: {PACKET_SIZE} bytes | Version: {VERSION}")
    print("Roundtrip, corruption, replay and stale-data tests: PASS")
    print(f"Sent={sent} | Simulated dropped={dropped} | Corrupted={corrupted}")
    print(f"Accepted={received} | Sequence rejects={receiver.rejected_replay}")
    print("Final stale-data rejection: PASS")
    print("Synthetic telemetry only; safety warnings NOT authorized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    test_protocol()
    simulate(count=args.count)
