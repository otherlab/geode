//#####################################################################
// Function popcount
//#####################################################################
#pragma once

#include <boost/static_assert.hpp>
#include <stdint.h>
namespace other {

#ifdef __GNUC__

static inline int popcount(uint16_t n) {
  return __builtin_popcount(n);
}

static inline int popcount(uint32_t n) {
  BOOST_STATIC_ASSERT(sizeof(int)==4);
  return __builtin_popcount(n);
}

static inline int popcount(uint64_t n) {
  BOOST_STATIC_ASSERT(sizeof(long)==8);
  return __builtin_popcountl(n);
}

#else
#error "Need to implement popcount for non-gcc compilers (using a parallel prefix circuit)"
#endif

}
