#pragma once
// Vector Dynamics protocol v1, little endian, 32 bytes. Desktop/embedded shared.
// No safety authorization; does not distinguish NO_TARGET from LANE_UNKNOWN.
#include <cstddef>
#include <cstdint>
namespace vd {
constexpr size_t PACKET_SIZE=32;
constexpr uint8_t VERSION=1;
constexpr uint8_t RANGE_VALID=1, VELOCITY_VALID=2, BEARING_VALID=4;
constexpr uint8_t CAMERA_FRESH=8, RADAR_FRESH=16, SYNTHETIC=32;
struct Packet {
 uint8_t source=0,destination=0,status=0,flags=0;
 uint32_t sequence=0,sender_ms=0;
 uint16_t track_id=0;
 int32_t range_mm=0,radial_mmps=0;
 int16_t bearing_centideg=0;
};
inline uint16_t u16(const uint8_t*p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
inline uint32_t u32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
inline void put16(uint8_t*p,uint16_t x){p[0]=uint8_t(x);p[1]=uint8_t(x>>8);}
inline void put32(uint8_t*p,uint32_t x){for(int i=0;i<4;i++)p[i]=uint8_t(x>>(8*i));}
inline uint32_t crc32(const uint8_t*data,size_t len){uint32_t c=0xffffffffu;for(size_t i=0;i<len;i++){c^=data[i];for(int j=0;j<8;j++)c=(c>>1)^(0xedb88320u&uint32_t(-(int32_t(c&1))));}return ~c;}
inline bool decode(const uint8_t*data,size_t n,Packet&out){
 if(n!=PACKET_SIZE||!data||data[0]!='V'||data[1]!='D'||data[2]!=VERSION||data[27]!=0)return false;
 if(crc32(data,28)!=u32(data+28))return false;
 if(data[3]<1||data[3]>3||data[4]<1||data[4]>3||data[26]>3||(data[25]&~uint8_t(63)))return false;
 Packet p;
 p.source=data[3];p.destination=data[4];p.sequence=u32(data+5);p.sender_ms=u32(data+9);
 p.track_id=u16(data+13);p.range_mm=int32_t(u32(data+15));p.radial_mmps=int32_t(u32(data+19));
 p.bearing_centideg=int16_t(u16(data+23));p.flags=data[25];p.status=data[26];
 if((p.flags&RANGE_VALID)&&p.range_mm<0)return false;
 if((p.flags&BEARING_VALID)&&(p.bearing_centideg < -18000||p.bearing_centideg > 18000))return false;
 out=p;return true;
}
inline bool encode(const Packet&p,uint8_t*out,size_t n){
 if(!out||n<PACKET_SIZE||p.source<1||p.source>3||p.destination<1||p.destination>3||p.status>3||(p.flags&~uint8_t(63)))return false;
 if((p.flags&RANGE_VALID)&&p.range_mm<0)return false;
 if((p.flags&BEARING_VALID)&&(p.bearing_centideg < -18000||p.bearing_centideg > 18000))return false;
 out[0]='V';out[1]='D';out[2]=VERSION;out[3]=p.source;out[4]=p.destination;
 put32(out+5,p.sequence);put32(out+9,p.sender_ms);put16(out+13,p.track_id);
 put32(out+15,uint32_t(p.range_mm));put32(out+19,uint32_t(p.radial_mmps));put16(out+23,uint16_t(p.bearing_centideg));
 out[25]=p.flags;out[26]=p.status;out[27]=0;put32(out+28,crc32(out,28));return true;
}
class Receiver {
 uint32_t last_sequence_[4]={};bool has_sequence_[4]={};
 uint32_t last_arrival_[4]={};bool has_packet_[4]={};Packet packets_[4]={};
public:
 bool ingest(const uint8_t*raw,size_t n,uint32_t now_ms){
  Packet p;if(!decode(raw,n,p)||p.destination!=3)return false;
  const unsigned s=p.source;
  if(has_sequence_[s]){uint32_t d=p.sequence-last_sequence_[s];if(d==0||d>=0x80000000u)return false;}
  has_sequence_[s]=true;last_sequence_[s]=p.sequence;last_arrival_[s]=now_ms;
  packets_[s]=p;has_packet_[s]=true;return true;
 }
 bool current(uint8_t source,uint32_t now_ms,Packet&out,uint32_t max_age_ms=300)const{
  if(source<1||source>3||!has_packet_[source]||now_ms-last_arrival_[source]>max_age_ms||packets_[source].status==3)return false;
  out=packets_[source];return true;
 }
 // A sender reboot requires an authenticated session-reset mechanism; not implemented.
};
}
