//#####################################################################
// Function sign
//#####################################################################
//
// finds the sign as +1, -1, or 0
//
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline T sign(const T a) {
  if (a>0) return 1;
  else if (a<0) return -1;
  else return 0;
}

template<class T> static inline T sign_nonzero(const T a) {
  return a>=0?1:-1;
}

}
