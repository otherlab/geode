// SSE helper routines
#pragma once

#include <geode/math/copysign.h>
#include <geode/math/isfinite.h>
#include <geode/utility/type_traits.h>
#include <iostream>

#ifdef __SSE__
#include <xmmintrin.h>
#include <emmintrin.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
namespace geode {

// Declaring these is legal on Windows, and they already exist for clang/gcc.
#ifdef _WIN32
static inline __m128 operator+(__m128 a, __m128 b) { return _mm_add_ps(a,b); }
static inline __m128 operator-(__m128 a, __m128 b) { return _mm_sub_ps(a,b); }
static inline __m128 operator*(__m128 a, __m128 b) { return _mm_mul_ps(a,b); }
static inline __m128 operator/(__m128 a, __m128 b) { return _mm_div_ps(a,b); }
static inline __m128i operator&(__m128i a, __m128i b) { return _mm_and_si128(a,b); }
static inline __m128i operator^(__m128i a, __m128i b) { return _mm_xor_si128(a,b); }
static inline __m128i operator|(__m128i a, __m128i b) { return _mm_or_si128(a,b); }
static inline __m128i operator~(__m128i a) { return _mm_andnot_si128(a,_mm_set1_epi32(~0)); }
static inline __m128 operator-(__m128 a) { return _mm_castsi128_ps(_mm_castps_si128(a)^_mm_set1_epi32(1<<31)); }
#endif

// Mark __m128 and __m128i as fundamental types
} namespace GEODE_TYPE_TRAITS_NAMESPACE {
template<> struct is_fundamental<__m128> : public geode::mpl::true_ {};
template<> struct is_fundamental<__m128i> : public geode::mpl::true_ {};
} namespace geode {

// fast_select(a,b,0) = a, fast_select(a,b,0xffffffff) = b, and anything else is undefined
static inline __m128i fast_select(__m128i a, __m128i b, __m128i mask) {
  // From http://markplusplus.wordpress.com/2007/03/14/fast-sse-select-operation
  return a^(mask&(a^b));
}

static inline __m128 fast_select(__m128 a, __m128 b, __m128i mask) {
  return _mm_castsi128_ps(fast_select(_mm_castps_si128(a),_mm_castps_si128(b),mask));
}

static inline __m128d fast_select(__m128d a, __m128d b, __m128i mask) {
  return _mm_castsi128_pd(fast_select(_mm_castpd_si128(a),_mm_castpd_si128(b),mask));
}

// Convenience version of fast_select
template<class T> static inline T sse_if(__m128i mask, T a, T b) {
  return fast_select(b,a,mask);
}
template<class T> static inline T sse_if(__m128d mask, T a, T b) {
  return fast_select(b,a,_mm_castpd_si128(mask));
}

inline __m128 min(__m128 a, __m128 b) {
  return _mm_min_ps(a,b);
}

inline __m128d min(__m128d a, __m128d b) {
  return _mm_min_pd(a,b);
}

// This exist as a primitive in SSE4, but we do it ourselves to be SSE2 compatible
inline __m128i min(__m128i a, __m128i b) {
  return fast_select(a,b,_mm_cmpgt_epi32(a,b));
}

inline __m128 max(__m128 a, __m128 b) {
  return _mm_max_ps(a,b);
}

inline __m128d max(__m128d a, __m128d b) {
  return _mm_max_pd(a,b);
}

template<class T> struct pack_type;
template<> struct pack_type<float>{typedef __m128 type;};
template<> struct pack_type<double>{typedef __m128d type;};
template<> struct pack_type<int32_t>{typedef __m128i type;};
template<> struct pack_type<int64_t>{typedef __m128i type;};
template<> struct pack_type<uint32_t>{typedef __m128i type;};
template<> struct pack_type<uint64_t>{typedef __m128i type;};

// Same as _mm_set_ps, but without the bizarre reversed ordering
template<class T> static inline typename pack_type<T>::type pack(T x0, T x1);
template<class T> static inline typename pack_type<T>::type pack(T x0, T x1, T x2, T x3);

template<> inline __m128d pack<double>(double x0, double x1) {
  return _mm_set_pd(x1,x0);
}

template<> inline __m128 pack<float>(float x0, float x1, float x2, float x3) {
  return _mm_set_ps(x3,x2,x1,x0);
}

template<> inline __m128i pack<int32_t>(int32_t x0, int32_t x1, int32_t x2, int32_t x3) {
  return _mm_set_epi32(x3,x2,x1,x0);
}

template<> inline __m128i pack<uint32_t>(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) {
  return _mm_set_epi32(x3,x2,x1,x0);
}

#if !defined(_WIN32) || defined(_WIN64)
template<> inline __m128i pack<int64_t>(int64_t x0, int64_t x1) {
  return _mm_set_epi64x(x1,x0);
}

template<> inline __m128i pack<uint64_t>(uint64_t x0, uint64_t x1) {
  return _mm_set_epi64x(x1,x0);
}
#endif

template<class D,class S> static inline D expand(S x);

template<> inline float expand(float x) {
  return x;
}

template<> inline double expand(double x) {
  return x;
}

template<> inline __m128 expand(float x) {
  return _mm_set_ps1(x);
}

template<> inline __m128d expand(double x) {
  return _mm_set1_pd(x);
}

static inline __m128 sqrt(__m128 a) {
  return _mm_sqrt_ps(a);
}

static inline __m128d sqrt(__m128d a) {
  return _mm_sqrt_pd(a);
}

#ifdef __SSE4_1__
static inline __m128 ceil(__m128 a) {
  return _mm_ceil_ps(a);
}

static inline __m128d ceil(__m128d a) {
  return _mm_ceil_pd(a);
}
#endif

static inline __m128 abs(__m128 a) {
  return _mm_castsi128_ps(_mm_castps_si128(a)&_mm_set1_epi32(~(1<<31)));
}

static inline __m128i isnotfinite(__m128 a) {
  const __m128i exponent = _mm_set1_epi32(0xff<<23);
  return _mm_cmpeq_epi32(_mm_castps_si128(a)&exponent,exponent);
}

static inline __m128i isfinite(__m128 a) {
  return ~isnotfinite(a);
}

static inline __m128 copysign(__m128 mag, __m128 sign) {
  return _mm_castsi128_ps((_mm_castps_si128(mag)&_mm_set1_epi32(~(1<<31)))|(_mm_castps_si128(sign)&_mm_set1_epi32(1<<31)));
}

static inline std::ostream& operator<<(std::ostream& os, __m128 a) {
  GEODE_ALIGNED(16) float x[4];
  _mm_store_ps(x,a);
  return os<<'['<<x[0]<<','<<x[1]<<','<<x[2]<<','<<x[3]<<']';
}

static inline std::ostream& operator<<(std::ostream& os, __m128i a) {
  int x[4];
  *(__m128i*)x = a;
  return os<<'['<<x[0]<<','<<x[1]<<','<<x[2]<<','<<x[3]<<']';
}

static inline void transpose(__m128i& i0, __m128i& i1, __m128i& i2, __m128i& i3) {
  __m128 f0 = _mm_castsi128_ps(i0),
         f1 = _mm_castsi128_ps(i1),
         f2 = _mm_castsi128_ps(i2),
         f3 = _mm_castsi128_ps(i3);
  _MM_TRANSPOSE4_PS(f0,f1,f2,f3);
  i0 = _mm_castps_si128(f0);
  i1 = _mm_castps_si128(f1);
  i2 = _mm_castps_si128(f2);
  i3 = _mm_castps_si128(f3);
}

}
#include <geode/vector/Vector.h>
namespace geode {

static inline Vector<double,2> unpack(const __m128d x) {
  return Vector<double,2>(_mm_cvtsd_f64(x),_mm_cvtsd_f64(_mm_unpackhi_pd(x,x)));
}

}
#endif
