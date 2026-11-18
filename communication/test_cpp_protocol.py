"""Cross-language check against the existing communication/vehicle_protocol.py.
Run from VectorDynamics root: python communication/test_cpp_protocol.py
Requires g++ on PATH; builds a temporary native executable.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "communication"))
from vehicle_protocol import (VehiclePacket, Module, Status, encode, decode,
    RANGE_VALID, VELOCITY_VALID, BEARING_VALID, RADAR_FRESH, SYNTHETIC)

CPP = ROOT / "firmware" / "common"

def run():
    packet = VehiclePacket(Module.FRONT, Module.CAB, 123, 1000, 42,
        60000, -4500, 1200,
        RANGE_VALID | VELOCITY_VALID | BEARING_VALID | RADAR_FRESH | SYNTHETIC,
        Status.CONFIRMED)
    raw = encode(packet)
    assert len(raw) == 32
    with tempfile.TemporaryDirectory() as temp:
        binary = str(Path(temp) / ("protocol_cli.exe" if sys.platform == "win32" else "protocol_cli"))
        subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-O2",
            str(CPP / "vehicle_protocol.cpp"), str(CPP / "protocol_cli.cpp"),
            "-o", binary], check=True)
        sample = subprocess.check_output([binary, "--sample"], text=True).strip()
        assert sample == raw.hex(), f"C++ vs Python bytes differ:\n{sample}\n{raw.hex()}"
        decoded = subprocess.check_output([binary, "--decode", raw.hex()], text=True).strip()
        expected = "1,3,123,1000,42,60000,-4500,1200,55,1"
        assert decoded == expected, (decoded, expected)
        assert decode(bytes.fromhex(sample)) == packet
        bad = bytearray(raw)
        bad[10] ^= 1
        check = subprocess.run([binary, "--decode", bad.hex()], capture_output=True, text=True)
        assert check.returncode != 0 and "BAD_CRC" in check.stderr
        print("C++ / Python 32-byte packet match: PASS")
        print("Signed fields and offsets: PASS")
        print("C++ corruption rejection: PASS")
        print("Golden packet:", raw.hex())
        print("Synthetic telemetry only; no safety warnings authorized")

if __name__ == "__main__":
    run()
