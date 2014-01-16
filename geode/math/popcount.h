//#####################################################################
// Function popcount
//#####################################################################
#pragma once

#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#endif
namespace geode {

#ifndef _WIN32

static inline int popcount(uint16_t n) {
  return __builtin_popcount(n);
}

static inline int popcount(uint32_t n) {
  static_assert(sizeof(int)==4,"");
  return __builtin_popcount(n);
}

static inline int popcount(uint64_t n) {
  static_assert(sizeof(long)==8,"");
  return __builtin_popcountl(n);
}

#else

static inline int popcount(uint16_t n) {
  return __popcnt16(n);
}

static inline int popcount(uint32_t n) {
  static_assert(sizeof(int)==4,"");
  return __popcnt(n);
}

static inline int popcount(uint64_t n) {
#ifdef _WIN64
  return (int)__popcnt64(n);
#else
  return __popcnt(uint32_t(n>>32))+__popcnt(uint32_t(n));
#endif
}

#endif

}
