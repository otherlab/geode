//#####################################################################
// Function min
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <algorithm>
#include <xmmintrin.h>
namespace other {

#undef min
using ::std::min;

// Forward declarations required by clang
inline __m128 min(__m128 a, __m128 b);
inline __m128i min(__m128i a, __m128i b);

template<class T> inline T min(const T a,const T b,const T c) {
  return min(a,min(b,c));
}

template<class T> inline T min(const T a,const T b,const T c,const T d) {
  return min(min(a,b),min(c,d));
}

template<class T> inline T min(const T a,const T b,const T c,const T d,const T e) {
  return min(min(a,b),min(c,d,e));
}

template<class T> inline T min(const T a,const T b,const T c,const T d,const T e,const T f) {
  return min(min(a,b,c),min(d,e,f));
}

template<class T> inline T min(const T a,const T b,const T c,const T d,const T e,const T f,const T g) {
  return min(min(a,b,c),min(d,e,f,g));
}

template<class T> inline T min(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h) {
  return min(min(a,b,c,d),min(e,f,g,h));
}

template<class T> inline T min(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h,const T i) {
  return min(min(a,b,c,d),min(e,f,g,h,i));
  }

static inline real min(real a,int b) {
  return min(a,(real)b);
}

static inline real min(int a,real b) {
  return min((real)a,b);
}

}
