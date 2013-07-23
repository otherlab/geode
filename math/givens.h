// Givens rotations
#pragma once

#include <other/core/vector/Vector.h>
namespace other {

// Compute a Givens rotation stored as a (cos,sin) pair, applying the rotation to the arguments in the process. 
template<class T> static inline Vector<T,2> givens_and_apply(T& x, T& y) {
  const T s = sqrt(x*x+y*y);
  const auto g = s ? Vector<T,2>(x,y)/s
                   : Vector<T,2>(1,0);
  x = s;
  y = 0;
  return g;
}

template<class T> static inline void givens_apply(const Vector<T,2>& g, T& x, T& y) {
  T tx = g.x*x+g.y*y;
  y = g.x*y-g.y*x;
  x = tx;
}

template<class T> static inline void givens_unapply(const Vector<T,2>& g, T& x, T& y) {
  T tx = g.x*x-g.y*y;
  y = g.x*y+g.y*x;
  x = tx;
}

}
