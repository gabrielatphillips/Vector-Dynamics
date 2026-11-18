#pragma once
#include <cstddef>
#include <cstdint>

namespace vd {
constexpr size_t PACKET_SIZE = 32;
constexpr size_t PAYLOAD_SIZE = 28;
enum Module : uint8_t { FRONT=1, REAR=2, CAB=3 };
enum Status : uint8_t { TENTATIVE=0, CONFIRMED=1, COASTING=2, EXPIRED=3 };
enum Flags : uint8_t { RANGE_VALID=1, VELOCITY_VALID=2, BEARING_VALID=4, CAMERA_FRESH=8, RADAR_FRESH=16, SYNTHETIC=32 };
struct VehiclePacket {
 uint8_t source, destination;
 uint32_t sequence, sender_ms;
 uint16_t track_id;
 int32_t range_mm, radial_mmps;
 int16_t bearing_centideg;
 uint8_t flags, status;
};
enum Error { OK=0, BAD_LENGTH, BAD_CRC, BAD_HEADER, BAD_FIELDS };
uint32_t crc32(const uint8_t* data, size_t length);
Error encode(const VehiclePacket& packet, uint8_t* out, size_t capacity);
Error decode(const uint8_t* data, size_t length, VehiclePacket& out);
const char* error_name(Error error);
}
