//#####################################################################
// Header integer_log
//#####################################################################
//
// Base 2 logarithms of integers and related functions.  Inexact versions round down.
//
//#####################################################################
#pragma once
#include <geode/config.h>
#include <cassert>
#include <stdint.h>
namespace geode {

inline int integer_log(uint32_t v) { // this works for any v, but it is slower
  int c=0;
  if(v&0xffff0000){v>>=16;c|=16;}
  if(v&0xff00){v>>=8;c|=8;}
  if(v&0xf0){v>>=4;c|=4;}
  if(v&0xc){v>>=2;c|=2;}
  if(v&2)c|=1;
  return c;
}

static inline int integer_log(int32_t v) {
  assert(v>=0);
  return integer_log(uint32_t(v));
}

inline int integer_log(uint64_t v) { // this works for any v, but it is slower
  int c=0;
  if(v&0xffffffff00000000){v>>=32;c|=32;}
  if(v&0xffff0000){v>>=16;c|=16;}
  if(v&0xff00){v>>=8;c|=8;}
  if(v&0xf0){v>>=4;c|=4;}
  if(v&0xc){v>>=2;c|=2;}
  if(v&2)c|=1;
  return c;
}

static inline int integer_log(int64_t v) {
  assert(v>=0);
  return integer_log(uint64_t(v));
}

static inline bool power_of_two(const uint32_t v) {
  return v>0 && (v&(v-1))==0;
}

static inline bool power_of_two(const uint64_t v) {
  return v>0 && (v&(v-1))==0;
}

inline uint32_t next_power_of_two(uint32_t v) {
  static_assert(sizeof(v)==4,"");v--;
  v|=v>>1;v|=v>>2;v|=v>>4;v|=v>>8;v|=v>>16;
  return v+1;
}

inline uint64_t next_power_of_two(uint64_t v) {
  static_assert(sizeof(v)==8,"");v--;
  v|=v>>1;v|=v>>2;v|=v>>4;v|=v>>8;v|=v>>16;v|=v>>32;
  return v+1;
}

static inline uint16_t min_bit(uint16_t v) {
  return v&(uint16_t)-(int16_t)v;
}

static inline uint32_t min_bit(uint32_t v) {
  return v&(uint32_t)-(int32_t)v;
}

static inline uint64_t min_bit(uint64_t v) {
  return v&(uint64_t)-(int64_t)v;
}

inline int integer_log_exact(const uint32_t v) { // this only works if v is a power of 2
  int log_value=(((v&0xffff0000)!=0)<<4)+(((v&0xff00ff00)!=0)<<3)+(((v&0xf0f0f0f0)!=0)<<2)+(((v&0xcccccccc)!=0)<<1)+((v&0xaaaaaaaa)!=0);
  assert(v==(uint32_t)1<<log_value);
  return log_value;
}

inline int integer_log_exact(const uint64_t v) { // this only works if v is a power of 2
  int log_value=(((v&0xffffffff00000000)!=0)<<5)+(((v&0xffff0000ffff0000)!=0)<<4)+(((v&0xff00ff00ff00ff00)!=0)<<3)+(((v&0xf0f0f0f0f0f0f0f0)!=0)<<2)+(((v&0xcccccccccccccccc)!=0)<<1)+((v&0xaaaaaaaaaaaaaaaa)!=0);
  assert(v==(uint64_t)1<<log_value);
  return log_value;
}

}
