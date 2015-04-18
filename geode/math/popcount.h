//#####################################################################
// Function popcount
//#####################################################################
#pragma once
#include <geode/config.h>
#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#endif
namespace geode {

#ifdef __GNUC__

static inline int popcount(uint16_t n) {
  return __builtin_popcount(n);
}

static inline int popcount(uint32_t n) {
  static_assert(sizeof(int)==4,"");
  return __builtin_popcount(n);
}

static inline int popcount(uint64_t n) {
#if __SIZEOF_LONG__ == 8
  static_assert(sizeof(long)==8,"");
  return __builtin_popcountl(n);
#elif __SIZEOF_LONG_LONG__ == 8
  static_assert(sizeof(long long)==8,"");
  return __builtin_popcountll(n);
#else
  #error "Can't deduce __builtin_popcount for uint64_t"
#endif

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
