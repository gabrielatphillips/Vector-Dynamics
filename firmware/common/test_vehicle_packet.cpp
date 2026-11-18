#include "vehicle_packet.h"
#include <cassert>
#include <cstdio>
int main(){
 vd::Packet p;p.source=2;p.destination=3;p.sequence=42;p.sender_ms=1000;p.track_id=7;
 p.range_mm=20000;p.radial_mmps=-2500;p.bearing_centideg=-300;
 p.flags=vd::RANGE_VALID|vd::VELOCITY_VALID|vd::BEARING_VALID|vd::SYNTHETIC;p.status=1;
 uint8_t b[32];assert(vd::encode(p,b,sizeof b));vd::Packet q;assert(vd::decode(b,sizeof b,q));
 assert(q.range_mm==20000&&q.radial_mmps==-2500&&q.sequence==42);
 vd::Receiver rx;assert(rx.ingest(b,32,5000));assert(!rx.ingest(b,32,5001));
 assert(rx.current(2,5299,q));assert(!rx.current(2,5301,q));
 b[10]^=1;assert(!vd::decode(b,32,q));
 std::puts("C++ vehicle packet: roundtrip/CRC/replay/stale PASS");
}
