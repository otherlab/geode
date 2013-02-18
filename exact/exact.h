// Multiprecision integer arithmetic for exact geometric predicates
#pragma once 

#include <other/core/exact/config.h>
#include <other/core/math/uint128.h>
namespace other {
namespace exact {

template<int b> struct IntBytes;
template<> struct IntBytes<4>  { typedef int32_t type; };
template<> struct IntBytes<8>  { typedef int64_t type; };
template<> struct IntBytes<12> { typedef __int128_t type; }; // Round up to 16 bytes
template<> struct IntBytes<16> { typedef __int128_t type; };

// Multiply two integers, returning a large enough integer to always hold the result
template<class I0,class I1> OTHER_ALWAYS_INLINE static inline typename IntBytes<sizeof(I0)+sizeof(I1)>::type emul(I0 x, I1 y) {
  return typename IntBytes<sizeof(I0)+sizeof(I1)>::type(x)*y;
}

template<class I> OTHER_ALWAYS_INLINE static inline typename IntBytes<2*sizeof(I)>::type esqr(I x) {
  BOOST_MPL_ASSERT((boost::is_same<I,typename IntBytes<sizeof(I)>::type>));
  return typename IntBytes<2*sizeof(I)>::type(x)*x;
}

template<class I> OTHER_ALWAYS_INLINE static inline typename IntBytes<3*sizeof(I)>::type ecube(I x) {
  return typename IntBytes<3*sizeof(I)>::type(esqr(x))*x;
}

// Interval versions of esqr and ecube for template purposes
OTHER_ALWAYS_INLINE static inline Interval esqr (Interval x) { return sqr(x); }
OTHER_ALWAYS_INLINE static inline Interval ecube(Interval x) { return cube(x); }

}
}
