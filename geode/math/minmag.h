//#####################################################################
// Function minmag
//#####################################################################
//
// finds the minimum value in magnitude and returns it with the sign
//
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline T minmag(const T a,const T b) {
  return abs(a)<abs(b)?a:b;
}

template<class T> static inline T minmag(const T a,const T b,const T c) {
  return minmag(a,minmag(b,c));
}

}
