#include "vehicle_protocol.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
static bool parse_hex(const std::string& s,uint8_t out[32]){
 if(s.size()!=64)return false;
 for(int i=0;i<32;i++){unsigned x; if(std::sscanf(s.substr(i*2,2).c_str(),"%2x",&x)!=1)return false;out[i]=uint8_t(x);}return true;
}
int main(int argc,char** argv){
 if(argc==2&&std::strcmp(argv[1],"--sample")==0){
  vd::VehiclePacket p{vd::FRONT,vd::CAB,123,1000,42,60000,-4500,1200,uint8_t(vd::RANGE_VALID|vd::VELOCITY_VALID|vd::BEARING_VALID|vd::RADAR_FRESH|vd::SYNTHETIC),vd::CONFIRMED};
  uint8_t raw[32];auto e=vd::encode(p,raw,sizeof raw);if(e!=vd::OK)return 1;
  for(uint8_t b:raw) std::printf("%02x",b);
  std::puts("");
  return 0;
 }
 if(argc==3&&std::strcmp(argv[1],"--decode")==0){
  uint8_t raw[32];if(!parse_hex(argv[2],raw)){std::cerr<<"invalid hex\n";return 2;}
  vd::VehiclePacket p{};auto e=vd::decode(raw,32,p);if(e!=vd::OK){std::cerr<<vd::error_name(e)<<"\n";return 3;}
  std::cout<<unsigned(p.source)<<","<<unsigned(p.destination)<<","<<p.sequence<<","<<p.sender_ms<<","<<p.track_id<<","<<p.range_mm<<","<<p.radial_mmps<<","<<p.bearing_centideg<<","<<unsigned(p.flags)<<","<<unsigned(p.status)<<"\n";return 0;
 }
 std::cerr<<"usage: protocol_cli --sample | --decode HEX\n";return 2;
}
