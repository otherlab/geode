//#####################################################################
// Function argmax
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline int argmax(const T a, const T b) {
  return a>=b?0:1;
}

template<class T> static inline int argmax(const T a, const T b, const T c) {
  if (a>=c) return argmax(a,b);
  return b>=c?1:2;
}

template<class T> inline int argmax(const T a, const T b, const T c, const T d) {
  T m = a;
  int r = 0;
  if (m<b) {m=b;r=1;}
  if (m<c) {m=c;r=2;}
  if (m<d) r=3;
  return r;
}

}
