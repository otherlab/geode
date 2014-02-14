// Utilities for endian conversion
#pragma once

#include <geode/utility/config.h>
#include <boost/integer.hpp>
#include <boost/detail/endian.hpp>
namespace geode {

// Handle unsigned ints specially

static inline uint8_t flip_endian(uint8_t x) {
  return x;
}

static inline uint16_t flip_endian(uint16_t x) {
  return x<<8|x>>8;
}

static inline uint32_t flip_endian(uint32_t x) {
  const uint32_t lo = 0x00ff00ff;
  x = (x&lo)<<8|(x>>8&lo);
  x = x<<16|x>>16;
  return x;
}

static inline uint64_t flip_endian(uint64_t x) {
  const uint64_t lo1 = 0x00ff00ff00ff00ff,
                 lo2 = 0x0000ffff0000ffff;
  x = (x&lo1)<<8|(x>>8&lo1);
  x = (x&lo2)<<16|(x>>16&lo2);
  x = x<<32|x>>32;
  return x;
}

// Vectors flip componentwise

template<class T,int d> static inline Vector<T,d> flip_endian(const Vector<T,d>& v) {
  Vector<T,d> r;
  for (int i=0;i<d;i++)
    r[i] = flip_endian(v[i]);
}

// For everything else, use the int case

template<class T> static inline T flip_endian(const T x) {
  const int n = sizeof(T);
  static_assert((n&(n-1))==0,"Size not a power of two");
  typedef typename boost::uint_t<8*n>::exact I;
  union {
    T x;
    I i;
  } u;
  u.x = x;
  u.i = flip_endian(u.i);
  return u.x;
}

// Convert to little endian
#if defined(BOOST_LITTLE_ENDIAN)
template<class T> static inline const T& to_little_endian(const T& x) { return x; }
#elif defined(BOOST_BIG_ENDIAN)
template<class T> static inline T to_little_endian(const T& x) { return flip_endian(x); }
#endif

}
