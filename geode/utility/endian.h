// Endianness detections and conversion
#pragma once

#include <geode/utility/config.h>
#ifdef __APPLE__
#include <sys/types.h>
#else
#include <endian.h>
#endif
namespace geode {

// How to detect endianness:
//
// #if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
// if (GEODE_ENDIAN == GEODE_LITTLE_ENDIAN)

#define GEODE_LITTLE_ENDIAN 1
#define GEODE_BIG_ENDIAN 2

#ifdef __APPLE__
#  if BYTE_ORDER == LITTLE_ENDIAN
#    define GEODE_ENDIAN GEODE_LITTLE_ENDIAN
#  elif BYTE_ORDER == BIG_ENDIAN
#    define GEODE_ENDIAN GEODE_BIG_ENDIAN
#  else
#    error Unknown machine endianness
#  endif
#else
#  if __BYTE_ORDER == __LITTLE_ENDIAN
#    define GEODE_ENDIAN GEODE_LITTLE_ENDIAN
#  elif __BYTE_ORDER == __BIG_ENDIAN
#    define GEODE_ENDIAN GEODE_BIG_ENDIAN
#  else
#    error Unknown machine endianness
#  endif
#endif

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
  typedef typename uint_t<8*n>::exact I;
  union {
    T x;
    I i;
  } u;
  u.x = x;
  u.i = flip_endian(u.i);
  return u.x;
}

// Convert to big or little endian
#if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
template<class T> static inline const T& to_little_endian(const T& x) { return x; }
template<class T> static inline       T  to_big_endian   (const T& x) { return flip_endian(x); }
#elif GEODE_ENDIAN == GEODE_BIG_ENDIAN
template<class T> static inline       T  to_little_endian(const T& x) { return flip_endian(x); }
template<class T> static inline const T& to_big_endian   (const T& x) { return x; }
#endif

// Same as to_, but useful for documentation purposes
template<class T> static inline T from_little_endian(const T& x) { return to_little_endian(x); }
template<class T> static inline T from_big_endian   (const T& x) { return to_big_endian(x); }

}
