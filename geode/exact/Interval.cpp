// Conservative interval arithmetic

#include <geode/exact/Interval.h>
#include <geode/exact/scope.h>
#include <geode/exact/Exact.h>
#include <geode/random/Random.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cout;
using std::endl;

namespace {
template<int a> struct ExactInterval {
  typedef ExactInterval Self;

  Exact<a> lo,hi;

  Self operator+(Self x) const {
    return Self({lo+x.lo,hi+x.hi});
  }

  Self operator-(Self x) const {
    return Self({lo-x.hi,hi-x.lo});
  }

  Self operator-() const {
    return Self({-hi,-lo});
  }

  ExactInterval<2*a> operator*(Self x) const {
    const auto e = lo*x.lo, b = lo*x.hi, c = hi*x.lo, d = hi*x.hi;
    return ExactInterval<2*a>({min(e,b,c,d),max(e,b,c,d)});
  }

  friend ExactInterval<2*a> sqr(Self x) {
    const auto lo = sqr(x.lo), hi = sqr(x.hi);
    return ExactInterval<2*a>({weak_sign(x)?min(lo,hi):Exact<2*a>(),max(lo,hi)});
  }

  Self operator<<(int s) const {
    return Self({lo<<s,hi<<s});
  }

  friend int weak_sign(Self x) {
    return sign(x.lo)<0 ? -1 : sign(x.hi)>0 ? 1 : 0;
  }

  friend Self min(Self x, Self y) {
    return Self({min(x.lo,y.lo),min(x.hi,y.hi)});
  }

  friend Self max(Self x, Self y) {
    return Self({max(x.lo,y.lo),max(x.hi,y.hi)});
  }

  friend ostream& operator<<(ostream& output, Self x) {
    return output << '[' << x.lo << ',' << x.hi << ']';
  }
};
}

static Tuple<ExactInterval<1>,Interval> random_interval(Random& random) {
  static_assert(sizeof(ExactInt)==8,"");
  const auto big = int64_t(1)<<52;
  auto lo = random.uniform<int64_t>(-big,big),
       hi = random.uniform<int64_t>(-big,big);
  if (!random.uniform<int>(0,5))
    lo = 0;
  if (!random.uniform<int>(0,5))
    hi = 0;
  if (lo > hi)
    lo = hi = (lo+hi)/2;
  return tuple(ExactInterval<1>({Exact<1>(lo),Exact<1>(hi)}),
               Interval(double(lo),double(hi)));
}

template<int a> static void to_mpf(mpf_t x, Exact<a> e) {
  // Turn e into an mpz_t
  const bool eneg = is_negative(e);
  if (eneg)
    e = -e;
  mpz_t n;
  mpz_init(n);
  mpz_import(n,a,-1,sizeof(mp_limb_t),0,0,e.n);
  if (eneg)
    mpz_neg(n,n);

  // Turn n into an mpz_f
  mpf_init(x);
  mpf_set_z(x,n);
  mpz_clear(n);
}

template<int a> static bool contains(const Interval i, const Exact<a> e) {
  mpf_t x;
  to_mpf(x,e);
  const auto b = i.box();
  const bool good = mpf_cmp_d(x,b.min)>=0 && mpf_cmp_d(x,b.max)<=0;
  mpf_clear(x);
  return good;
}

template<int a> static bool contains(const Interval i, const ExactInterval<a> e) {
  return contains(i,e.lo) && contains(i,e.hi);
}

static bool equal(const Interval i, const ExactInterval<1> e) {
  mpf_t lo,hi;
  to_mpf(lo,e.lo);
  to_mpf(hi,e.hi);
  const auto b = i.box();
  const bool good = mpf_cmp_d(lo,b.min)==0 && mpf_cmp_d(hi,b.max)==0;
  mpf_clear(lo);
  mpf_clear(hi);
  return good;
}

static inline bool implies(const bool a, const bool b) {
  return !a || b;
}

void interval_tests(const int steps) {
  cout << "GEODE_INTERVAL_SSE = "<<GEODE_INTERVAL_SSE<<endl;
  IntervalScope scope;
  const auto random = new_<Random>(1218131);
  for (int step=0;step<steps;step++) {
    // Make two random intervals
    const auto p0 = random_interval(random),
               p1 = random_interval(random);
    const auto e0 = p0.x, e1 = p1.x;
    const auto i0 = p0.y, i1 = p1.y;
    GEODE_ASSERT(contains(i0,e0));
    GEODE_ASSERT(contains(i1,e1));

    // Verify that most arithmetic is conservative.
    // Warning: We do not currently check operator>>, inverse, and assume_safe_sqrt.
    GEODE_ASSERT(contains(i0+i1,e0+e1));
    GEODE_ASSERT(contains(i0-i1,e0-e1));
    GEODE_ASSERT(contains(i0*i1,e0*e1));
    GEODE_ASSERT(contains(-i0,-e0));
    GEODE_ASSERT(contains(sqr(i0),sqr(e0)));
    for (int s=0;s<=2;s++)
      GEODE_ASSERT(contains(i0<<s,e0<<s));
    GEODE_ASSERT(equal(min(i0,i1),min(e0,e1)));
    GEODE_ASSERT(equal(max(i0,i1),max(e0,e1)));

    // Check sign routines
    const int s0 = weak_sign(e0),
              s1 = weak_sign(e1);
    GEODE_ASSERT(implies(certainly_negative(i0),s0<0));
    GEODE_ASSERT(implies(certainly_positive(i0),s0>0));
    GEODE_ASSERT(implies(certainly_zero(i0),sign(e0.lo)==0 && sign(e0.hi)==0));
    GEODE_ASSERT(implies(certainly_opposite_sign(i0,i1),s0*s1<0));
    GEODE_ASSERT(implies(certainly_less(i0,i1),weak_sign(e1-e0)>0));
  }
}

}
