//#####################################################################
// Function max
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <algorithm>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
namespace geode {

#undef max
using ::std::max;

// Forward declarations required by clang
#ifdef __SSE__
inline __m128 max(__m128 a, __m128 b);
inline __m128i max(__m128i a, __m128i b);
#endif

template<class T> inline T max(const T a,const T b,const T c) {
  return max(a,max(b,c));
}

template<class T> inline T max(const T a,const T b,const T c,const T d) {
  return max(max(a,b),max(c,d));
}

template<class T> inline T max(const T a,const T b,const T c,const T d,const T e) {
  return max(max(a,b),max(c,d,e));
}

template<class T> inline T max(const T a,const T b,const T c,const T d,const T e,const T f) {
  return max(max(a,b,c),max(d,e,f));
}

template<class T> inline T max(const T a,const T b,const T c,const T d,const T e,const T f,const T g) {
  return max(max(a,b,c),max(d,e,f,g));
}

template<class T> inline T max(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h) {
  return max(max(a,b,c,d),max(e,f,g,h));
}

template<class T> inline T max(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h,const T i) {
  return max(max(a,b,c,d),max(e,f,g,h,i));
}

static inline real max(real a,int b) {
  return max(a,(real)b);
}

static inline real max(int a,real b) {
  return max((real)a,b);
}

}
