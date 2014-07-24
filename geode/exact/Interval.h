// Conservative interval arithmetic
#pragma once

// Modified from code by Tyson Brochu, 2011

// We use double precision arithmetic unconditionally.  This is particularly important for constructions,
// where we want to error range to be better than float precision in most cases.  If possible, we use SSE
// to accelerate interval arithmetic and avoid incorrect code with clang.

#include <geode/exact/config.h>
#include <geode/geometry/Box.h>
#include <geode/math/constants.h>
#include <geode/math/sse.h>
#include <geode/python/repr.h>
#include <geode/utility/rounding.h>
namespace geode {

struct Interval;
struct IntervalScope;
using std::ostream;

// IMPORTANT: All interval arithmetic must occur within an IntervalScope (see scope.h).

// Intervals are treated as scalars so that they are preserved through various arithmetic operations
template<> struct IsScalar<Interval> : public mpl::true_ {};

// Use packed hashing
template<> struct is_packed_pod<Interval> : public mpl::true_ {};

// If possible, use SSE to speed up interval arithmetic.
#if defined(__SSE4_1__)
#define GEODE_INTERVAL_SSE 1
#else
#define GEODE_INTERVAL_SSE 0
#endif

#if !GEODE_INTERVAL_SSE

struct Interval {
  typedef IntervalScope Scope;

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

  Box<double> box() const {
    return Box<double>(-nlo,hi);
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

static inline Interval min(const Interval& a, const Interval& b) {
  Interval result;
  result.hi = min(a.hi, b.hi);
  result.nlo = max(a.nlo, b.nlo); // result.lo = min(a.lo, a.lo)
  return result;
}
static inline Interval max(const Interval& a, const Interval& b) {
  Interval result;
  result.hi = max(a.hi, b.hi);
  result.nlo = min(a.nlo, b.nlo); // result.lo = max(a.lo, a.lo)
  return result;
}

static inline Interval operator-(const double x, const Interval y) {
  assert(fegetround() == FE_UPWARD);
  return Interval(-(y.hi-x),x+y.nlo);
}

static inline Interval operator+(const double x, const Interval y) {
  return y+x;
}

// Unfortunately, clang does not understand rounding modes, and assumes that x * -y = -(x * y).
// Therefore, we implement our own version of negation to hide this identity from clang, taking
// advantage of the fact that negation is flipping the high bit.  Unfortunately, this makes the
// motion planning code 6% slower on Forrest's machine, but it is important for correctness.
// Note that this version of the code applies only without SSE.
static inline double safe_neg(const double x) {
#if defined(__clang__)
  union { double x; uint64_t i; } u;
  u.x = x;
  u.i = u.i ^ (uint64_t(1)<<63);
  return u.x;
#else
  return -x;
#endif
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
      r.nlo = b * safe_neg(d);
      r.hi  = na * nc;
    } else if (0 <= nc) {
      r.nlo = na * d;
      r.hi  = na * nc;
    } else {
      r.nlo = na * d;
      r.hi  = b * safe_neg(nc);
    }
  } else if (0 <= na) {
    if (d <= 0) {
      r.nlo = b * nc;
      r.hi  = na * nc;
    } else if (0 <= nc) {
      r.nlo = max(na * d, b * nc);
      r.hi  = max(na * nc, b * d);
    } else {
      r.nlo = na * d;
      r.hi  = b * d;
    }
  } else {
    if (d <= 0) {
      r.nlo = b * nc;
      r.hi  = na * safe_neg(d);
    } else if (0 <= nc) {
      r.nlo = b * nc;
      r.hi  = b * d;
    } else {
      r.nlo = na * safe_neg(nc);
      r.hi  = b * d;
    }
  }

  assert(!(-r.nlo > r.hi)); // Use 'not greater' instead of 'less than or equal' so that nan won't trigger this
  return r;
}

static inline Interval sqr(const Interval x) {
  assert(fegetround() == FE_UPWARD);
  Interval s;
  if (x.nlo < 0) { // x > 0
    s.nlo = x.nlo * safe_neg(x.nlo);
    s.hi = x.hi * x.hi;
  } else if (x.hi < 0) { // x < 0
    s.nlo = x.hi * safe_neg(x.hi);
    s.hi = x.nlo * x.nlo;
  } else { // 0 in x
    s.nlo = 0;
    s.hi = sqr(max(x.nlo,x.hi));
  }
  assert(-s.nlo <= s.hi);
  return s;
}

// Provide shifts as special functions since they're exact
GEODE_ALWAYS_INLINE static inline Interval operator<<(const Interval x, const int p) {
  assert(unsigned(p)<32);
  const double y = 1<<p;
  Interval s;
  s.nlo = y*x.nlo;
  s.hi  = y*x.hi;
  return s;
}
GEODE_ALWAYS_INLINE static inline Interval operator>>(const Interval x, const int p) {
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
  // For the lower bound, we multiply by 1-4ep < (1+e)(1+e)^-2 to avoid switching rounding modes.
  // The factor of (1+e)^-2 handles the upwards rounding of sqrt, and the factor of (1+e) handles
  // the upwards rounding of the multiplication.  The latter is required because clang replaces
  // -(tweak*x) with (-tweak)*x.
  const double tweak = 4*numeric_limits<double>::epsilon()-1;
  s.nlo = x.nlo>=0 ? 0 : -sqrt(tweak*x.nlo);
  assert(-s.nlo <= s.hi);
  return s;
}

static inline string repr(const Interval x) {
  return format("[%s,%s]",repr(-x.nlo),repr(x.hi));
}

static inline ostream& operator<<(ostream& output, const Interval x) {
  const double c = x.center();
  return output << c << "+-" << x.hi-c;
}

// Are all intervals in a vector strictly smaller than threshold?
template<int m> static inline bool small(const Vector<Interval,m>& xs, const double threshold) {
  for (auto& x : xs)
    if (x.nlo+x.hi >= threshold)
      return false;
  return true;
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

static inline Interval abs(const Interval x) {
  if (x.nlo > 0) {
    if (x.hi > 0) // x.nlo > 0 && x.hi > 0
      return Interval(0, max(x.nlo, x.hi));
    else // x.nlo > 0 && x.hi <= 0
      return Interval(-x.hi, x.nlo);
  } else
    return x;
}

#else // Faster SSE versions!

struct Interval {
  typedef IntervalScope Scope;

  typedef double T;

  // In SSE mode, we pack the two doubles into a single __m128d.
  // As above, s is (-nlo,hi).
  __m128d s;

  Interval()
    : s(_mm_setzero_pd()) {}

  Interval(T x)
    : s(pack(-x,x)) {}

  Interval(T lo, T hi)
    : s(pack(-lo,hi)) {
    assert(lo <= hi);
  }

  // Warning: The __m128d must be in (-lo,hi) form already.
  explicit Interval(__m128d s)
    : s(s) {}

  static Interval full() {
    return Interval(-inf,inf);
  }

  bool contains(const Interval x) const {
    // -nlo <= -x.nlo && x.hi <= hi;
    // x.nlo <= nlo && x.hi <= hi;
    return _mm_movemask_epi8(_mm_castpd_si128(_mm_cmple_pd(x.s,s))) == 0xffff;
  }

  bool contains_zero() const {
    return contains(Interval());
  }

  bool contains(T x) const {
    return contains(Interval(x));
  }

  Interval& operator+=(const Interval x) {
    assert(fegetround() == FE_UPWARD);
    s += x.s;
    return *this;
  }

  Interval& operator-=(const Interval x) {
    return *this += -x;
  }

  Interval& operator*=(const Interval x) {
    return *this = *this*x;
  }

  Interval operator+(const Interval x) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(s+x.s);
  }

  Interval operator-(const Interval x) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(s+(-x).s);
  }

  Interval operator*(const Interval x) const;

  Interval operator-() const {
    return Interval(_mm_shuffle_pd(s,s,1));
  }

  // The best estimate of the actual value
  double center() const {
    const auto b = box();
    return .5*(b.max+b.min);
  }

  Interval thickened(T delta) const {
    assert(fegetround() == FE_UPWARD);
    return Interval(s+expand<__m128d>(delta));
  }

  Box<double> box() const {
#if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
    union { __m128d s; struct { double nlo,hi; } d; } u;
#elif GEODE_ENDIAN == GEODE_BIG_ENDIAN
    union { __m128d s; struct { double hi,nlo; } d; } u;
#endif
    u.s = s;
    return Box<double>(-u.d.nlo,u.d.hi);
  }
};

static inline bool certainly_negative(const Interval x) {
  // x.hi < 0
  return _mm_movemask_epi8(_mm_castpd_si128(_mm_cmplt_pd(x.s,_mm_setzero_pd()))) & 0x0100;
}

static inline bool certainly_positive(const Interval x) {
  // x.nlo < 0
  return _mm_movemask_epi8(_mm_castpd_si128(_mm_cmplt_pd(x.s,_mm_setzero_pd()))) & 0x0001;
}

static inline bool certainly_zero(const Interval x) {
  // !x.nlo && !x.hi
  return _mm_movemask_epi8(_mm_castpd_si128(_mm_cmpeq_pd(x.s,_mm_setzero_pd()))) == 0xffff;
}

static inline bool certainly_opposite_sign(const Interval a, const Interval b) {
  return (certainly_negative(a) && certainly_positive(b))
      || (certainly_positive(a) && certainly_negative(b));
}

static inline bool certainly_less(const Interval a, const Interval b) {
  // a.hi < -b.nlo
  return _mm_movemask_epi8(_mm_castpd_si128(_mm_cmplt_pd(_mm_shuffle_pd(a.s,a.s,1),-b.s))) & 1;
}

static inline Interval min(const Interval a, const Interval b) {
  return Interval(_mm_shuffle_pd(max(a.s,b.s),min(a.s,b.s),2));
}

static inline Interval max(const Interval a, const Interval b) {
  return Interval(_mm_shuffle_pd(min(a.s,b.s),max(a.s,b.s),2));
}

static inline Interval operator-(const double x, const Interval y) {
  return Interval(x)-y;
}

static inline Interval operator+(const double x, const Interval y) {
  return Interval(x)+y;
}

// As above, we hide negation from clang to prevent incorrect code.
static inline __m128d safe_neg(const __m128d x) {
#ifdef __clang__
  return _mm_castsi128_pd(_mm_castpd_si128(x) ^ _mm_castpd_si128(expand<__m128d>(-0.)));
#else
  return -x;
#endif
}

static inline __m128d safe_neg_lo(const __m128d x) {
  return _mm_castsi128_pd(_mm_castpd_si128(x) ^ _mm_castpd_si128(pack(-0.,0.)));
}

inline Interval Interval::operator*(const Interval x) const {
  // Given intervals [a,b] and [c,d], the product is
  //   P = [min(ac,ad,bc,bd),max(ac,ad,bc,bd)]
  //     = [min(a_c,a_d,b_c,b_d),max(a^c,a^d,b^c,b^d)]  # _ means * down, ^ means * up
  //     = [-max(na^-nc,na^d,b^nc,b^-d),max(na^nc,na^-d,b^-nc,b^d)]
  //     = [-max(na^d,b^nc,na^-nc,b^-d),max(na^nc,b^d,na^-d,b^-nc)]
  // We implement this branch free, computing all the products using four SSE
  // multiplications and then reducing using max.
  const auto fx = _mm_shuffle_pd(x.s,x.s,1),
             ns = safe_neg(s),
             u0 = s*x.s,  // [na*nc,b*d]   - hi,hi
             u1 = s*fx,   // [na*d,b*nc]   - lo,lo
             u2 = ns*x.s, // [na*-nc,b*-d] - lo,lo
             u3 = ns*fx;  // [na*-d,b*-nc] - hi,hi
  // Transpose u so that each product goes on the right side of the __m128d.
  const auto v0 = _mm_shuffle_pd(u1,u0,0),
             v1 = _mm_shuffle_pd(u1,u0,3),
             v2 = _mm_shuffle_pd(u2,u3,0),
             v3 = _mm_shuffle_pd(u2,u3,3);
  // And we're done!
  return Interval(max(max(v0,v1),max(v2,v3)));
}

static inline Interval sqr(const Interval x) {
  assert(fegetround() == FE_UPWARD);
  const auto zero = _mm_setzero_pd(),
             s = x.s,
             f = _mm_shuffle_pd(s,s,1),
             neg = _mm_cmplt_pd(s,zero),
             neglo = _mm_shuffle_pd(neg,neg,0),
             neghi = _mm_shuffle_pd(neg,neg,3),
             u = sse_if(neglo,s,sse_if(neghi,f,_mm_shuffle_pd(zero,max(s,f),0)));
  return Interval(u*safe_neg_lo(u));
}

// Provide shifts as special functions since they're exact
GEODE_ALWAYS_INLINE static inline Interval operator<<(const Interval x, const int p) {
  assert(unsigned(p)<32);
  const double y = 1<<p;
  return Interval(x.s*expand<__m128d>(y));
}
GEODE_ALWAYS_INLINE static inline Interval operator>>(const Interval x, const int p) {
  assert(unsigned(p)<32);
  const double y = 1./(1<<p);
  return Interval(x.s*expand<__m128d>(y));
}

// Valid only for intervals that don't contain zero
static inline Interval inverse(const Interval x) {
  assert(!x.contains_zero() && fegetround() == FE_UPWARD);
  return Interval(expand<__m128d>(-1.)/_mm_shuffle_pd(x.s,x.s,1));
}

// Take the sqrt root of an interval, assuming the true input is nonnegative.
// For explanation, see the non-SSE version above.
static inline Interval assume_safe_sqrt(const Interval x) {
  assert(fegetround()==FE_UPWARD && !certainly_negative(x));
  const auto tweak = pack(4*numeric_limits<double>::epsilon()-1,1.);
  return Interval(pack(-1.,1.)*sqrt(max(tweak*x.s,_mm_setzero_pd())));
}

static inline string repr(const Interval x) {
  const auto b = x.box();
  return format("[%s,%s]",repr(b.min),repr(b.max));
}

static inline ostream& operator<<(ostream& output, const Interval x) {
  const auto b = x.box();
  const double c = b.center();
  return output << c << "+-" << b.max-c;
}

// Are all intervals in a vector smaller than threshold?
template<int m> static inline bool small(const Vector<Interval,m>& xs, const double threshold) {
  const auto th = expand<__m128d>(threshold);
  __m128i bad = _mm_setzero_si128();
  for (int i=0;i<m;i++) {
    const auto s = xs[i].s;
    bad |= _mm_castpd_si128(_mm_cmpge_pd(s+_mm_shuffle_pd(s,s,1),th));
  }
  return !_mm_movemask_epi8(bad);
}

// Conservatively expand an integer to an integer box
template<int m> static inline Box<Vector<Quantized,m>> snap_box(const Vector<Interval,m>& xs) {
  Box<Vector<Quantized,m>> box;
  for (int i=0;i<m;i++) {
#ifdef __SSE4_1__
    const auto b = ceil(xs[i].s);
    box.min[i] = b.min;
    box.max[i] = b.max;
#else
    const auto b = xs[i].box();
    box.min[i] = floor(b.min);
    box.max[i] =  ceil(b.max);
#endif
  }
  return box;
}

static inline Interval abs(const Interval x) {
  const auto zero = _mm_setzero_pd(),
             flip = _mm_shuffle_pd(x.s,x.s,1), // hi,-lo
             hi = max(x.s,flip),               // max(-lo,hi) in both
             lo = max(-flip,zero),             // max(0,-hi)  in both
             y = _mm_shuffle_pd(lo,hi,0);      // abs(x) if xlo > 0
  // Trues if x.nlo > 0
  const auto flag = _mm_cmpgt_pd(_mm_shuffle_pd(x.s,x.s,0),zero);
  return Interval(sse_if(flag,y,x.s));
}

#endif // GEODE_INTERVAL_SSE

// +-1 if the interval is definitely nonzero, otherwise zero
static inline int weak_sign(const Interval x) {
  return certainly_positive(x) ?  1
       : certainly_negative(x) ? -1
                               :  0;
}

static inline bool overlap(const Interval x, const Interval y) {
  return !(certainly_less(x,y) || certainly_less(y,x));
}

// The center of an interval vector
template<int m> static inline Vector<double,m> center(const Vector<Interval,m>& xs) {
  Vector<double,m> c;
  for (int i=0;i<m;i++)
    c[i] = xs[i].center();
  return c;
}

// Snap an interval vector to integers, rounding to the best integer guess
template<int m> static inline Vector<Quantized,m> snap(const Vector<Interval,m>& xs) {
  Vector<Quantized,m> r;
  for (int i=0;i<m;i++)
    r[i] = Quantized(round(xs[i].center()));
  return r;
}

inline Box<Vec2> bounding_box(const Vector<Interval,2>& i) {
  const auto bx = i.x.box(), by = i.y.box();
  return Box<Vec2>(Vec2(bx.min,by.min),Vec2(bx.max,by.max));
}

} // namespace geode
