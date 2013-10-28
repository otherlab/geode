//#####################################################################
// Function minabs
//#####################################################################
//
// finds the minimum absolute value
//
//#####################################################################
#pragma once

#include <cmath>
#include <geode/math/min.h>
namespace geode {

// a should already be nonnegative
template<class T> static inline T minabs_incremental(const T a,const T b) {
  return min(a,abs(b));
}

template<class T> static inline T minabs(const T a,const T b) {
  return min(abs(a),abs(b));
}

template<class T> static inline T minabs(const T a,const T b,const T c) {
  return minabs_incremental(minabs(a,b),c);
}

template<class T> static inline T minabs(const T a,const T b,const T c,const T d) {
  return min(minabs(a,b),minabs(c,d));
}

template<class T> static inline T minabs(const T a,const T b,const T c,const T d,const T e) {
  return minabs_incremental(minabs(a,b,c,d),e);
}

template<class T> static inline T minabs(const T a,const T b,const T c,const T d,const T e,const T f) {
  return min(minabs(a,b,c,d),minabs(e,f));
}

}
