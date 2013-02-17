// Conservative interval arithmetic
#pragma once

// Modified from code by Tyson Brochu, 2011

#include <fenv.h>
namespace other {

// IMPORTANT: All interval arithmetic must occur within an IntervalScope (see scope.h).

struct Interval {
  // For now, we use single precision arithmetic unconditionally.
  typedef float T;
  typedef T Scalar;

  // We store the interval [a,b] as (-a,b) internally.  With proper arithmetic operations, this
  // allows us to use only FE_UPWARD and avoid switching rounding modes over and over.
  T nlo, hi;

  Interval()
    : nlo(0), hi(0) {}

  Interval(T x)
    : nlo(-x), hi(x) {}  

  Interval(T lo, T hi)
    : nlo(-lo), hi(hi) {
    assert(lo <= hi);
  }
  
  bool contains_zero() const {
    return nlo>=0 && hi>=0;
  }

  bool certainly_negative() const {
    return hi < 0;
  }

  bool certainly_positive() const {
    return nlo < 0;
  }

  bool certainly_zero() const {
    return !nlo && !hi;
  }

  bool contains(T x) const {
    return -nlo<=x && x<=hi;
  }

  Interval& operator+=(const Interval x) {
    assert(fegetround() == FE_UPWARD);
    nlo += x.nlo;
    hi += x.hi;
    return *this;
  }

  Interval& operator-=(const Interval x) {
    assert(fegetround() == FE_UPWARD);
    nlo += x.hi;
    hi += x.nlo;
    return *this;
  }

  Interval& operator*=(const Interval x) {
    return *this = *this*x;
  }

  Interval operator+(const Interval x) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(-(nlo+x.nlo),hi+x.hi);
  }
  
  Interval operator-(const Interval x) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(-(nlo+x.hi),hi+x.nlo);
  }
  
  Interval operator*(const Interval x) const;
  
  Interval operator-() const {
    return Interval(-hi,nlo);
  }
  
  static void begin_special_arithmetic();
  static void end_special_arithmetic();
};

static inline bool certainly_opposite_sign(const Interval a, const Interval b) {
  return (a.certainly_negative() && b.certainly_positive())
      || (a.certainly_positive() && b.certainly_negative());
}

inline Interval Interval::operator*(const Interval x) const {
  assert(fegetround() == FE_UPWARD);
  const T na = nlo,
          b = hi,
          nc = x.nlo,
          d = x.hi;
  Interval r;
  if (b <= 0) {
    if (d <= 0) {
      r.nlo = -b * d;
      r.hi  = na * nc;
    } else if (-nc <= 0 && 0 <= d) {
      r.nlo = na * d;
      r.hi  = na * nc;
    } else {
      r.nlo = na * d;
      r.hi  = b * -nc;
    }
  } else if (-na <= 0 && 0 <= b) {
    if (d <= 0) {
      r.nlo = b * nc;
      r.hi  = na * nc;
    } else if (-nc <= 0 && 0 <= d) {
      r.nlo = max(na * d, b * nc);
      r.hi  = max(na * nc, b * d);
    } else {
      r.nlo = na * d;
      r.hi  = b * d;
    }
  } else {
    if (d <= 0) {
      r.nlo = b * nc; 
      r.hi  = -na * d;
    } else if (-nc <= 0 && 0 <= d) {
      r.nlo = b * nc;
      r.hi  = b * d;
    } else {
      r.nlo = -na * nc;
      r.hi  = b * d;
    }
  }
  assert(-r.nlo <= r.hi);
  return r;
}

static inline Interval sqr(const Interval x) {
  assert(fegetround() == FE_UPWARD);
  Interval s;
  if (x.nlo < 0) { // x > 0
    s.nlo = x.nlo * -x.nlo;
    s.hi = x.hi * x.hi;
  } else if (x.hi < 0) { // x < 0
    s.nlo = x.hi * -x.hi;
    s.hi = x.nlo * x.nlo;
  } else { // 0 in x
    s.nlo = 0;
    s.hi = sqr(max(x.nlo,x.hi));
  }
  assert(-s.nlo <= s.hi);
  return s;
}

static inline Interval cube(const Interval x) {
  assert(fegetround() == FE_UPWARD);
  return Interval(-(x.nlo*x.nlo*x.nlo),x.hi*x.hi*x.hi);
}

}
