#include "vehicle_protocol.h"
namespace vd {
static void u16(uint8_t* p, uint16_t v){p[0]=uint8_t(v);p[1]=uint8_t(v>>8);}
static void u32(uint8_t* p, uint32_t v){for(int i=0;i<4;i++)p[i]=uint8_t(v>>(8*i));}
static uint16_t r16(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
static uint32_t r32(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
uint32_t crc32(const uint8_t* p,size_t n){uint32_t crc=0xffffffffu;for(size_t i=0;i<n;i++){crc^=p[i];for(int j=0;j<8;j++)crc=(crc>>1)^((crc&1)?0xedb88320u:0u);}return crc^0xffffffffu;}
static bool valid(const VehiclePacket& p){
 if(p.source<FRONT||p.source>CAB||p.destination<FRONT||p.destination>CAB||p.status>EXPIRED|| (p.flags&~uint8_t(63)))return false;
 if((p.flags&RANGE_VALID)&&p.range_mm<0)return false;
 if((p.flags&BEARING_VALID)&&(p.bearing_centideg< -18000||p.bearing_centideg>18000))return false;
 return true;
}
Error encode(const VehiclePacket& p,uint8_t* out,size_t n){
 if(!out||n<PACKET_SIZE)return BAD_LENGTH;
 if(!valid(p))return BAD_FIELDS;
 out[0]='V';out[1]='D';out[2]=1;out[3]=p.source;out[4]=p.destination;
 u32(out+5,p.sequence);u32(out+9,p.sender_ms);u16(out+13,p.track_id);
 u32(out+15,uint32_t(p.range_mm));u32(out+19,uint32_t(p.radial_mmps));u16(out+23,uint16_t(p.bearing_centideg));
 out[25]=p.flags;out[26]=p.status;out[27]=0;
 u32(out+28,crc32(out,PAYLOAD_SIZE));return OK;
}
Error decode(const uint8_t* data,size_t n,VehiclePacket& p){
 if(!data||n!=PACKET_SIZE)return BAD_LENGTH;
 if(crc32(data,PAYLOAD_SIZE)!=r32(data+28))return BAD_CRC;
 if(data[0]!='V'||data[1]!='D'||data[2]!=1||data[27]!=0)return BAD_HEADER;
 VehiclePacket temp{};temp.source=data[3];temp.destination=data[4];temp.sequence=r32(data+5);temp.sender_ms=r32(data+9);
 temp.track_id=r16(data+13);temp.range_mm=int32_t(r32(data+15));temp.radial_mmps=int32_t(r32(data+19));temp.bearing_centideg=int16_t(r16(data+23));temp.flags=data[25];temp.status=data[26];
 if(!valid(temp)) return BAD_FIELDS;
 p=temp;
 return OK;
}
const char* error_name(Error e){switch(e){case OK:return "OK";case BAD_LENGTH:return "BAD_LENGTH";case BAD_CRC:return "BAD_CRC";case BAD_HEADER:return "BAD_HEADER";default:return "BAD_FIELDS";}}
}
