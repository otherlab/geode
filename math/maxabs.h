//#####################################################################
// Function maxabs
//#####################################################################
//
// finds the maximum absolute value
//
//#####################################################################
#pragma once

#include <cmath>
#include <other/core/math/max.h>
namespace other {

using std::abs;

template<class T> inline T maxabs(const T a,const T b) {
  return max(abs(a),abs(b));
}

template<class T> inline T maxabs(const T a,const T b,const T c) {
  return max(maxabs(a,b),abs(c));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d) {
  return max(maxabs(a,b,c),abs(d));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d,const T e) {
  return max(maxabs(a,b,c,d),abs(e));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d,const T e,const T f) {
  return max(maxabs(a,b,c,d,e),abs(f));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d,const T e,const T f,const T g) {
  return max(maxabs(a,b,c,d,e,f),abs(g));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h) {
  return max(maxabs(a,b,c,d,e,f,g),abs(h));
}

template<class T> inline T maxabs(const T a,const T b,const T c,const T d,const T e,const T f,const T g,const T h,const T i) {
  return max(maxabs(a,b,c,d,e,f,g,h),abs(i));
}

}
