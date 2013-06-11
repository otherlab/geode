// Conservative interval arithmetic
#pragma once

// Modified from code by Tyson Brochu, 2011

#include <other/core/utility/config.h>
#include <other/core/geometry/forward.h>
#include <other/core/math/constants.h>
#include <other/core/python/repr.h>
#include <fenv.h>
namespace other {

struct Interval;
using std::ostream;

// IMPORTANT: All interval arithmetic must occur within an IntervalScope (see scope.h).

// Intervals are treated as scalars so that they are preserved through various arithmetic operations
template<> struct IsScalar<Interval> : public mpl::true_ {};

struct Interval {
  // For now, we use double precision arithmetic unconditionally.  This is particularly important for constructions,
  // where we want to error range to be better than float precision in most cases.
  typedef double T;

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

  static Interval full() {
    return Interval(-inf,inf);
  }

  bool contains_zero() const {
    return nlo>=0 && hi>=0;
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

  // The best estimate of the actual value
  double center() const {
    return .5*(hi-nlo);
  }

  Interval thickened(T delta) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(-(nlo+delta),hi+delta);
  }
};

static inline bool certainly_negative(const Interval x) {
  return x.hi < 0;
}

static inline bool certainly_positive(const Interval x) {
  return x.nlo < 0;
}

static inline bool certainly_zero(const Interval x) {
  return !x.nlo && !x.hi;
}

static inline bool certainly_opposite_sign(const Interval a, const Interval b) {
  return (certainly_negative(a) && certainly_positive(b))
      || (certainly_positive(a) && certainly_negative(b));
}

static inline bool certainly_less(const Interval a, const Interval b) {
  return a.hi < -b.nlo;
}

// +-1 if the interval is definitely nonzero, otherwise zero
static inline int weak_sign(const Interval x) {
  return certainly_positive(x) ?  1
       : certainly_negative(x) ? -1
                               :  0;
}

static inline Interval operator-(const double x, const Interval y) {
  assert(fegetround() == FE_UPWARD);
  return Interval(-(y.hi-x),x+y.nlo);
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

// Provide shifts as special functions since they're exact
OTHER_ALWAYS_INLINE static inline Interval operator<<(const Interval x, const int p) {
  assert(unsigned(p)<32);
  const double y = 1<<p;
  Interval s;
  s.nlo = y*x.nlo;
  s.hi  = y*x.hi;
  return s;
}
OTHER_ALWAYS_INLINE static inline Interval operator>>(const Interval x, const int p) {
  assert(unsigned(p)<32);
  const double y = 1./(1<<p);
  Interval s;
  s.nlo = y*x.nlo;
  s.hi  = y*x.hi;
  return s;
}

// Valid only for intervals that don't contain zero
static inline Interval inverse(const Interval x) {
  assert(!x.contains_zero() && fegetround() == FE_UPWARD);
  Interval s;
  s.nlo = -1/x.hi;
  s.hi  = -1/x.nlo;
  assert(-s.nlo <= s.hi);
  return s;
}

// Take the sqrt root of an interval, assuming the true input is nonnegative
static inline Interval assume_safe_sqrt(const Interval x) {
  assert(fegetround()==FE_UPWARD);
  assert(x.hi >= 0);
  Interval s;
  // The upper bound is easy
  s.hi = sqrt(x.hi);
  // For the lower bound, we multiply by 1-3ep < (1+e)^-2 (rounding towards zero in the multiplication) to avoid switching rounding modes.
  const double tweak = 1-3*numeric_limits<double>::epsilon();
  s.nlo = x.nlo>=0 ? 0 : -sqrt(-(tweak*x.nlo));
  assert(-s.nlo <= s.hi);
  return s;
}

static inline string repr(const Interval x) {
  return format("[%s,%s]",repr(-x.nlo),repr(x.hi));
}

static inline ostream& operator<<(ostream& output, const Interval x) {
  const double c = x.center();
  return output << x.center() << "+-" << x.hi-c;
}

// Are all intervals in a vector smaller than threshold?
template<int m> static inline bool small(const Vector<Interval,m>& xs, const double threshold) {
  for (auto& x : xs)
    if (x.nlo+x.hi >= threshold)
      return false;
  return true;
}

// Snap an interval vector to integers, rounding to the best integer guess
template<int m> static inline Vector<Quantized,m> snap(const Vector<Interval,m>& xs) {
  Vector<Quantized,m> r;
  for (int i=0;i<m;i++)
    r[i] = Quantized(round(xs[i].center()));
  return r;
}

// Conservatively expand an integer to an integer box
template<int m> static inline Box<Vector<Quantized,m>> snap_box(const Vector<Interval,m>& xs) {
  Box<Vector<Quantized,m>> box;
  for (int i=0;i<m;i++) {
    box.min[i] = Quantized(floor(-xs[i].nlo));
    box.max[i] = Quantized( ceil( xs[i].hi));
  }
  return box;
}

}
