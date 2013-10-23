//#####################################################################
// Function argmin
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline int argmin(const T a, const T b) {
  return b<a?1:0;
}

template<class T> static inline int argmin(const T a, const T b, const T c) {
  if (a<c) return b<a?1:0;
  return c<b?2:1;
}

template<class T> inline int argmin(const T a, const T b, const T c, const T d) {
  T m = a;
  int r = 0;
  if (m>b) {m=b;r=1;}
  if (m>c) {m=c;r=2;}
  if (m>d) r=3;
  return r;
}

}
