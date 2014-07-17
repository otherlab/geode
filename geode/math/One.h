//#####################################################################
// Class One
//#####################################################################
#pragma once

#include <geode/math/Zero.h>
#include <iostream>
namespace geode {

struct One {
  explicit operator bool() const {
    return true;
  }

  One operator*(const One) const {
    return One();
  }

  bool operator==(const One) const {
    return true;
  }

  One inverse() const {
    return One();
  }

  static One one() {
    return One();
  }

  // Make One usable as an expression functor in exact/perturb.h
  static const int degree = 0;
#ifdef GEODE_VARIADIC 
  template<class... Args> static One eval(const Args&... args) { return One(); }
#endif
};

template<class T> static inline const T& operator*(const T& x, const One) {
  return x;
}

template<class T> static inline const T& operator*(const One, const T& x) {
  return x;
}

template<class T> static inline const T& operator/(const T& x, const One) {
  return x;
}

template<class T> static inline T& operator*=(T& x, const One) {
  return x;
}

template<class T> static inline T& operator/=(T& x, const One) {
  return x;
}

static inline std::ostream& operator<<(std::ostream& output, const One) {
  return output<<'1';
}

}
