//#####################################################################
// Class Zero
//#####################################################################
#pragma once

#include <iostream>
namespace other {

struct Zero {
  bool operator!() const {
    return true;
  }

  Zero operator-() const {
    return Zero();
  }

  Zero operator-(const Zero) const {
    return Zero();
  }

  Zero operator+(const Zero) const {
    return Zero();
  }

  Zero operator*(const Zero) const {
    return Zero();
  }

  template<class T> Zero operator*=(const T&) {
    return Zero();
  }

  template<class T> Zero operator/=(const T&) {
    return Zero();
  }

  Zero operator+=(const Zero) {
    return Zero();
  }

  Zero operator-=(const Zero) {
    return Zero();
  }

  Zero inverse() const {
    return Zero();
  }

  bool operator==(const Zero) const {
    return true;
  }

  bool operator!=(const Zero) const {
    return false;
  }

  bool positive_semidefinite() const {
    return true;
  }
};

static inline bool operator<(const float x,const Zero) {
  return x<0;
}

static inline bool operator<(const double x,const Zero) {
  return x<0;
}

static inline bool operator>(const float x,const Zero) {
  return x>0;
}

static inline bool operator>(const double x,const Zero) {
  return x>0;
}

template<class T> static inline const T& operator+(const T& x, const Zero) {
  return x;
}

template<class T> static inline const T& operator+(const Zero, const T& x) {
  return x;
}

template<class T> static inline const T& operator-(const T& x,const Zero) {
  return x;
}

template<class T> static inline T operator-(const Zero, const T& x) {
  return -x;
}

template<class T> static inline Zero operator*(const Zero,const T&) {
  return Zero();
}

template<class T> static inline Zero operator*(const T&, const Zero) {
  return Zero();
}

static inline std::ostream& operator<<(std::ostream& output, const Zero) {
  return output<<'0';
}

}
