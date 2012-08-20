//#####################################################################
// Function givens_rotate
//#####################################################################
//
// Applies a Givens rotation to a pair of scalars
//
//#####################################################################
#pragma once

namespace other {

template<class T> inline void givens_rotate(T& x, T& y, const T c, const T s) {
  T w = c*x+s*y;
  y = c*y-s*x;
  x = w;
}

}
