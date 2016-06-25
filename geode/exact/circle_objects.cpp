#include <geode/exact/circle_objects.h>
#include <geode/exact/Exact.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/exact/predicates.h>
#include <cmath>
namespace geode {

////////////////////////////////////////////////////////////////////////////////

static inline exact::Perturbed<3> perturbed(const ExactCircle<Pb::Explicit> c) {
  return exact::Perturbed<3>(c.index,c.center,c.radius);
}
static inline exact::Perturbed<2> perturbed_center(const ExactCircle<Pb::Explicit> c) {
  return exact::Perturbed<2>(c.index, c.center);
}

static inline exact::ImplicitlyPerturbed<3> perturbed(const ExactCircle<Pb::Implicit> c) {
  return exact::ImplicitlyPerturbed<3>(c.center,c.radius);
}
static inline exact::ImplicitlyPerturbedCenter perturbed_center(const ExactCircle<Pb::Implicit> c) {
  return exact::ImplicitlyPerturbedCenter(c.center,c.radius);
}

// Perturbation adds dummy values to ExactHorizontals to treat it as a circles
// These values are choosen to ensure no collisions will occur with well formed circles
//   index = numeric_limits<int>::max()
//   center.x = 0
//   radius = -1
static inline exact::Perturbed<3> perturbed(const ExactHorizontal<Pb::Explicit>& h) {
  return exact::Perturbed<3>(numeric_limits<int>::max(),0,h.y,-1);
}
static inline exact::Perturbed<2> perturbed_center(const ExactHorizontal<Pb::Explicit>& h) {
  return exact::Perturbed<2>(numeric_limits<int>::max(),0,h.y);
}
static inline exact::ImplicitlyPerturbed<3> perturbed(const ExactHorizontal<Pb::Implicit>& h) {
    return exact::ImplicitlyPerturbed<3>(0,h.y,-1);
}
static inline exact::ImplicitlyPerturbedCenter perturbed_center(const ExactHorizontal<Pb::Implicit>& h) {
    return exact::ImplicitlyPerturbedCenter(0,h.y,-1);
}

////////////////////////////////////////////////////////////////////////////////

template<> bool is_same_circle(const ExactCircle<Pb::Implicit>& c0, const ExactCircle<Pb::Implicit>& c1) {
  return (c0.radius == c1.radius) && (c0.center == c1.center);
}
template<> bool is_same_circle(const ExactCircle<Pb::Explicit>& c0, const ExactCircle<Pb::Explicit>& c1) {
  assert(c0.index != c1.index || ((c0.radius == c1.radius) && (c0.center == c1.center)));
  return (c0.index == c1.index);
}

template<Pb PS> bool is_same_horizontal(const ExactHorizontal<PS>& h0, const ExactHorizontal<PS>& h1) {
  return h0.y == h1.y;
}

template<Pb PS> bool is_same_intersection(const CircleIntersectionKey<PS>& i0, const CircleIntersectionKey<PS>& i1) {
  return is_same_circle(i0.cl, i1.cl) && is_same_circle(i0.cr, i1.cr);
}

template<Pb PS> bool is_same_intersection(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1) {
  assert(is_same_horizontal(i0.line, i1.line));
  return (i0.left == i1.left) && is_same_circle(i0.circle, i1.circle);
}

// Do two circles intersect (degree 2)?
namespace {
template<bool add> struct Intersect {
  template<class TV> static inline PredicateType<2,TV> eval(const TV S0, const TV S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return sqr(add?r1+r0:r1-r0)-esqr_magnitude(c1-c0);
}};}
template<Pb PS> bool has_intersections(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1) {
  return     perturbed_predicate<Intersect<true >>(perturbed(c0),perturbed(c1))
         && !perturbed_predicate<Intersect<false>>(perturbed(c0),perturbed(c1));
}

template<Pb PS> bool circles_overlap(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1) {
  return perturbed_predicate<Intersect<true >>(perturbed(c0),perturbed(c1));
}

namespace {
struct Alpha { template<class TV> static PredicateType<2,TV> eval(const TV S0, const TV S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return esqr_magnitude(c1-c0)+(r0+r1)*(r0-r1);
}};
template<int i,int j> struct Beta { template<class... Args> static PredicateType<4,typename First<Args...>::type> eval(const Args... args) {
  const auto S = tuple(args...);
  const auto& Si = S.template get<i>();
  const auto& Sj = S.template get<j>();
  const auto c0 = Si.xy(), c1 = Sj.xy();
  const auto r0 = Si.z,    r1 = Sj.z;
  const auto sqr_dc = esqr_magnitude(c1-c0);
  return sqr(r0<<1)*sqr_dc-sqr(sqr_dc+(r0+r1)*(r0-r1));
}};
}

// Which quadrant is in the intersection of two circles in relative to the center of the first?
// The quadrants are 0 to 3 counterclockwise from positive/positive.
// This function should be used only from circle_circle_intersections, where it is precomputed as CircleIntersection::q
// As written this is degree 6, but it can be reduced to degree 2 if necessary.
namespace {
template<int axis> struct QuadrantA { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1) {
  return Alpha::eval(S0,S1)*(axis==0?S1.y-S0.y:S0.x-S1.x);
}};
template<int axis> struct QuadrantB { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1) {
  return S1[axis]-S0[axis]; // dc[axis]
}};}
template<Pb PS> uint8_t circle_circle_intersection_quadrant(const ReferenceSide side, const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ApproxIntersection v) {
  assert(circle_circle_approx_intersection(cl,cr).box().intersects(v.box()));
  // We must evaluate the predicate
  //
  //   cross(e,alpha*dc+beta*dc^perp) > 0
  //   alpha cross(e,dc) + beta dot(e,dc) > 0
  //
  // where e0 = (1,0) and e1 = (0,1).  This has the form
  //
  //   A + B sqrt(C) > 0
  //   A = alpha_hat cross(e,dc)
  //   B = dot(e,dc)
  //   C = beta_hat^2
  //
  // Compute predicates for both axes
  //
  const ExactCircle<PS>& c0 = cl_is_reference(side) ? cl : cr;
  const ExactCircle<PS>& c1 = cl_is_reference(side) ? cr : cl;
  const int s = cl_is_reference(side) ? -1 : 1;

  const bool p0 = FILTER(v.p().y-c0.center.y, perturbed_predicate_sqrt<QuadrantA<0>,QuadrantB<0>,Beta<0,1>>(s,perturbed(c0),perturbed(c1))),
             p1 = FILTER(c0.center.x-v.p().x, perturbed_predicate_sqrt<QuadrantA<1>,QuadrantB<1>,Beta<0,1>>(s,perturbed(c0),perturbed(c1)));
  // Assemble our two predicates into a quadrant
  return 2*!p0+(p0==p1);
}

namespace {
#define UPWARDS_PRELUDE() \
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy(); \
  const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z; \
  const auto c01 = c1-c0, c02 = c2-c0; \
  const auto sqr_c01 = esqr_magnitude(c01), \
             sqr_c02 = esqr_magnitude(c02); \
  const auto alpha01 = sqr_c01+(r0+r1)*(r0-r1), \
             alpha02 = sqr_c02+(r0+r2)*(r0-r2);
template<bool add> struct UpwardsA { template<class TV> static PredicateType<5,TV> eval(const TV S0, const TV S1, const TV S2) {
  UPWARDS_PRELUDE()
  const auto first  = alpha02*(c02.y*sqr_c01),
             second = alpha01*(c01.y*sqr_c02);
  return add ? first+second : first-second;
}};
template<int i> struct UpwardsB { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1, const TV S2) {
  static_assert(i==1 || i==2,"");
  const auto c01 = S1.xy()-S0.xy(),
             c02 = S2.xy()-S0.xy();
  return i==1 ? c01.x*esqr_magnitude(c02) // Negated below
              : c02.x*esqr_magnitude(c01);
}};
template<bool add,int i> struct UpwardsDE { template<class TV> static PredicateType<6,TV> eval(const TV S0, const TV S1, const TV S2) {
  UPWARDS_PRELUDE()
  // Happily, D/positive and the two factors of E/positive all have quite similar form, so we encode them into the same template here.  From below, the three expressions are
  //   D/positive =    c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02  c1y c2y            - 4 r0^2 ((c1y c2x)^2 + (c1x c2y)^2 + 2 (c1x c2x)^2)
  //   E/positive =   (c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02 (c1y c2y - c1x c2x) - 4 r0^2 (c1x c2y + c1y c2x)^2)
  //                * (c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02 (c1y c2y + c1x c2x) - 4 r0^2 (c1x c2y - c1y c2x)^2)
  // All three have the form
  //   second = alpha01*alpha02*2*F
  //   factor = first +- second - 4*sqr(r0)*G
  // Mapping D to i = 0 and E to i = 1,2, we get
  const auto c1x2x = c01.x*c02.x,
             c1y2y = c01.y*c02.y,
             c1x2y = c01.x*c02.y,
             c1y2x = c01.y*c02.x;
  const auto F = i==0 ? c1y2y
               : i==1 ? c1y2y-c1x2x
               :        c1y2y+c1x2x;
  const auto G = i==0 ? sqr(c1y2x)+sqr(c1x2y)+(sqr(c1x2x)<<1)
                      : sqr(i==1 ? c1x2y+c1y2x
                                 : c1x2y-c1y2x);
  const auto first = sqr_c02*sqr(alpha01)+sqr_c01*sqr(alpha02),
             second = alpha01*alpha02*(F<<1);
  return (add?first+second:first-second)-sqr(r0<<1)*G;
}};
template<bool add> struct UpwardsF { template<class TV> static PredicateType<8,TV> eval(const TV S0, const TV S1, const TV S2) {
  UPWARDS_PRELUDE()
  // F/positive = c01^2 (alpha02 (alpha02 c01^2 +- 2 alpha01 c1y c2y) + (2r0)^2 ((c1x c2y)^2 - (c1y c2x)^2)) - alpha01^2 (c1x^2 - c1y^2) c02^2
  const auto first  = alpha02*sqr_c01,
             second = (c01.y<<1)*c02.y*alpha01;
  return sqr_c01*(alpha02*(add?first+second:first-second)+sqr(r0<<1)*(sqr(c01.x*c02.y)-sqr(c01.y*c02.x)))-sqr(alpha01)*((sqr(c01.x)-sqr(c01.y))*sqr_c02);
}};}
template<bool add,class P3> GEODE_ALWAYS_INLINE static inline bool perturbed_upwards(const int sign1, const int sign2, const P3 S0, const P3 S1, const P3 S2) {
  // This routine is an optimized version of perturbed_predicate_two_sqrts specialized to this particular predicate, taking advantage of polynomial factorization
  // to reduce the degree from 20 to 8.  This improves on the degree 12 result of Devillers et al., Algebraic methods and arithmetic filtering for exact predicates on circle arcs,
  // which is possible since our predicate operates on three unique arcs instead of four.

  // Our predicate is a function of three arcs (c0,r0),(c1,r1),(c2,r2).  Let j = 3-i.  Defining
  //   si = signi
  //   c01 = c1-c0
  //   c02 = c2-c0
  //   alpha0i = c01^2+(r0+ri)(r0-ri) = c0i^2+r0^2-ri^2
  //   A = alpha02 (c02.y c01^2) +- alpha01 (c01.y c02^2)
  //   Bi = c0iy c0j^2
  //   Ci = 4r0^2 c0i^2 - alpha0i^2
  // our predicate is
  //   A + s1 B1 sqrt(C1) + s2 B2 sqrt(C2) > 0
  // Below, we will include si in Bi to get
  //   A + B1 sqrt(C1) + B2 sqrt(C2) > 0
  typedef UpwardsA<add> A; // Degree 5
  typedef UpwardsB<1> B1;  // Degree 3
  typedef UpwardsB<2> B2;
  GEODE_DEBUG_ONLY(typedef Beta<0,1> C1;) // Degree 4
  GEODE_DEBUG_ONLY(typedef Beta<0,2> C2;)

  // First, some consistency checks
  assert(abs(sign1)==1 && abs(sign2)==1);
  assert(perturbed_predicate<C1>(S0,S1,S2));
  assert(perturbed_predicate<C2>(S0,S1,S2));

  // As in the general case, we next check if all three terms have the same sign.  This is degree 5 due to A.
  const int sA  =        perturbed_predicate<A> (S0,S1,S2) ? 1 : -1,
            sB1 = sign1*(perturbed_predicate<B1>(S0,S1,S2) ? 1 : -1),
            sB2 = sign2*(perturbed_predicate<B2>(S0,S1,S2) ? 1 : -1);
  if (sA==sB1 && sA==sB2)
    return sA > 0;

  // We now have a choice of what to move to the RHS: B1 sqrt(C1), B2 sqrt(C2), or both.  In order to maximize
  // speed, we make this choice differently depending on the signs of A, B1, B2.  If B1 and B2 have the same
  // sign, moving both to the RHS turns out to require only degree 6 predicates, so we do that.  However, if
  // B1 and B2 have different signs, determining the sign of the RHS after moving both over requires a degree
  // 10 predicate.  Therefore, we instead move exactly the term which differs from A in sign, which requires
  // at most degree 8 predicates.

  // If B1 and B2 have the same sign, go the degree 6 route:
  int sign_flips;
  if (sB1 == sB2) {

    // Move *both* sqrt terms to the RHS and square once.  Moving both terms is different from perturbed_predicate_two_sqrts, but lets us reach 6 if sB1==sB2.
    // We use the notation s> to mean > if s>0 and < if s<0.
    //   A + B1 sqrt(C1) + B2 sqrt(C2) > 0
    //   A > -B1 sqrt(C1) - B2 sqrt(C2)
    //   A^2 sA> B1^2 C1 + B2^2 C2 + 2 B1 B2 sqrt(C1 C2)
    //   A^2 - B1^2 C1 - B2^2 C2 - 2 B1 B2 sqrt(C1 C2) sA> 0
    //   D - 2 B1 B2 sqrt(C1 C2) sA> 0
    // where
    //   D = A^2 - B1^2 C1 - B2^2 C2
    // D is degree 10 but is a multiple of c01^2 c02^2 as shown in Mathematica.  Removing these unconditionally positive factors and simplifying produces
    //   D/positive = c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02 c1y c2y - 4 r0^2 ((c1y c2x)^2 + (c1x c2y)^2 + 2 (c1x c2x)^2)
    typedef UpwardsDE<add,0> D;
    const int sD = perturbed_predicate<D>(S0,S1,S2) ? 1 : -1;
    if (sD==-sB1*sB2)
      return sD*sA > 0;

    // Now we square once more to get our final polynomial:
    //   D - 2 B1 B2 sqrt(C1 C2) sA> 0
    //   D sA> 2 B1 B2 sqrt(C1 C2)
    //   D^2 sAsD> 4 B1^2 B2^2 C1 C2
    //   D^2 - 4 B1^2 B2^2 C1 C2 sAsD> 0
    //   E sAsD> 0
    // where
    //   E = D^2 - 4 B1^2 B2^2 C1 C2
    // is degree 20.  E factors into c01^2 c02^2 and two degree 6 factors:
    //   E/positive =   (c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02 (c1y c2y - c1x c2x) - 4 r0^2 (c1x c2y + c1y c2x)^2)
    //                * (c02^2 alpha01^2 + c01^2 alpha02^2 +- 2 alpha01 alpha02 (c1y c2y + c1x c2x) - 4 r0^2 (c1x c2y - c1y c2x)^2)
    sign_flips = sA*sD;

  } else { // sB1 != sB2

    // Define i,j by sA == sBi, sA != sBj.  We have
    //   A + Bi sqrt(Ci) + Bj sqrt(Cj) > 0
    //   A + Bi sqrt(Ci) > -Bj sqrt(Cj)
    //   A^2 + Bi^2 Ci + 2 A Bi sqrt(Ci) sA> Bj^2 Cj
    //   A^2 + Bi^2 Ci - Bj^2 Cj sA> -2 A Bi sqrt(Ci)
    //   F sA> -2 A Bi sqrt(Ci)
    // where
    //   F = A^2 + Bi^2 Ci - Bj^2 Cj
    // F has degree 10, but is a multiple of c0j^2, so reduces to degree 8.  If i=1,j=2, we have
    //   F/positive = c01^2 (alpha02 (alpha02 c01^2 +- 2 alpha01 c1y c2y) + (2r0)^2 ((c1x c2y)^2 - (c1y c2x)^2)) - alpha01^2 (c1x^2 - c1y^2) c02^2
    typedef UpwardsF<add> F;
    const int i = sA==sB1 ? 1 : 2;
    const int sF = perturbed_predicate<F>(S0,i==1?S1:S2,i==1?S2:S1) ? 1 : -1;
    if (sF == 1)
      return sF*sA > 0;

    // As before, we square once more to get our final polynomial
    //   F sA> -2 A Bi sqrt(Ci)
    //   F^2 sAsF> 4 A^2 Bi^2 Ci
    //   F^2 - 4 A^2 Bi^2 Ci sAsF> 0
    //   E sAsF> 0
    // since we have
    //   E = F^2 - 4 A^2 Bi^2 Ci = D^2 - 4 B1^2 B2^2 C1 C2 // The formula for E from above
    sign_flips = sA*sF;
  }

  // The sB1 == sB2 and sB1 != sB2 join up here, since they both use the same E predicate.
  typedef UpwardsDE<add,1> Ea;
  typedef UpwardsDE<add,2> Eb;
  const int sE =    perturbed_predicate<Ea>(S0,S1,S2)
                 == perturbed_predicate<Eb>(S0,S1,S2) ? 1 : -1;
  return sE*sign_flips > 0;
}

template<Pb PS> static inline bool circle_intersections_upwards_degenerate(const ExactCircle<PS>& c0, const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) {
  if(is_same_circle(i0, i1)) {
    // If we are looking at the left and right intersections from a pair of circles...
    assert(!c0.is_same_intersection(i0,i1)); // Verify that we aren't being asked to compare an intersections with itself
    return rightwards(perturbed_center(c0),perturbed_center(i0)) ^ cl_is_incident(i0.side);
  }
  else {
    return perturbed_upwards<false>(cl_is_incident(i0.side)?-1:1,cl_is_incident(i1.side)?1:-1,perturbed(c0),perturbed(i0),perturbed(i1));
  }
}

namespace {
template<int sign> struct CircleIntersectsHorizontal { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1) {
  const auto cy = S0.y,
             r = S0.z,
             y = S1.y;
  return sign>0 ? (cy+r)-y
                : y-(cy-r);
}};}
template<Pb PS> bool has_intersections(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1) {
  return FILTER((Interval(c0.center.y)+c0.radius)-h1.y,
                perturbed_predicate<CircleIntersectsHorizontal<+1>>(perturbed(c0),perturbed(h1)))
      && FILTER(h1.y-(Interval(c0.center.y)-c0.radius),
                perturbed_predicate<CircleIntersectsHorizontal<-1>>(perturbed(c0),perturbed(h1)));
}

#if GEODE_KEEP_INTERSECTION_INTERVALS
static Vector<Interval,2> approx_interval(const exact::Vec2 snapped) {
  return Vector<Interval,2>(Interval(snapped.x-ApproxIntersection::tolerance(),snapped.x+ApproxIntersection::tolerance()),
                            Interval(snapped.y-ApproxIntersection::tolerance(),snapped.y+ApproxIntersection::tolerance()));
}
#endif

// Construct both of the intersections of two circular arcs, assuming they do intersect.
// result.x will be CircleIntersection(circle0, circle1).approx
// result.y will be CircleIntersection(circle1, circle0).approx
// The results differ from the true intersections by at most 1.
// Degrees 3/2 for the nonsqrt part and 6/4 for the part under the sqrt.
template<Pb PS> static Vector<ApproxIntersection,2> circle_circle_approx_intersections(const ExactCircle<PS>& circle0, const ExactCircle<PS>& circle1) {
  assert(!is_same_circle(circle0, circle1)); // Shouldn't be calling this on a circle with itself
  assert(has_intersections(circle0, circle1));
  Vector<ApproxIntersection,2> v;
#if CHECK
  GEODE_WARNING("Expensive consistency checking enabled");
  Vector<Interval,2> check_linear, check_quadratic;
  check_linear.fill(Interval::full());
  check_quadratic.fill(Interval::full());
#endif

  // Evaluate conservatively using intervals
  {
    const Vector<Interval,2> c0(circle0.center), c1(circle1.center);
    const Interval           r0(circle0.radius), r1(circle1.radius);
    const auto dc = c1-c0;
    const Interval sqr_dc = sqr_magnitude(dc);
    if (certainly_positive(sqr_dc)) {
      const auto half_inv_sqr_dc = inverse(sqr_dc<<1),
                 sqr_r0 = sqr(r0),
                 alpha_hat = sqr_dc-sqr(r1)+sqr_r0;
      const auto linear = c0+half_inv_sqr_dc*alpha_hat*dc;
#if CHECK
      check_linear = linear;
#endif
      if (small(linear,.5)) {
        const auto beta_hat = assume_safe_sqrt((sqr_r0*(sqr_dc<<2))-sqr(alpha_hat));
        const auto quadratic = half_inv_sqr_dc*beta_hat*rotate_left_90(dc);
#if CHECK
        check_quadratic = quadratic;
#endif
        if (small(quadratic,.5) && !CHECK) {
#if GEODE_KEEP_INTERSECTION_INTERVALS
          v.x._approx_interval = linear-quadratic;
          v.y._approx_interval = linear+quadratic;
#else
          const auto sl = snap(linear),
                     sq = snap(quadratic);
          v.x._rounded = sl-sq;
          v.y._rounded = sl+sq;
#endif
          return v;
        }
      }
    }
  }

  {
    // If intervals fail, evaluate and round using symbolic perturbation.  For simplicity, we round the sqrt part
    // separately from the rational part, at the cost of a maximum error of 1 (1/2+1/2).  The full formula is
    //
    //   X = FR +- perp(sqrt(FS))
    #define MOST \
      const Vector<Exact<1>,2> c0(X[0].xy()), c1(X[1].xy()); \
      const Exact<1> r0(X[0].z), r1(X[1].z); \
      const auto dc = c1-c0; \
      const auto sqr_dc = esqr_magnitude(dc), \
                 two_sqr_dc = sqr_dc<<1, \
                 alpha_hat = sqr_dc-(r1+r0)*(r1-r0); \
      assert(result.m==3);
    struct FR { static void eval(RawArray<mp_limb_t,2> result, RawArray<const Vector<Exact<1>,3>> X) {
      MOST
      const auto v = emul(two_sqr_dc,c0)+emul(alpha_hat,dc);
      mpz_set(result[0],v.x);
      mpz_set(result[1],v.y);
      mpz_set(result[2],two_sqr_dc);
    }};
    struct FS { static void eval(RawArray<mp_limb_t,2> result, RawArray<const Vector<Exact<1>,3>> X) {
      MOST
      const auto sqr_beta_hat = sqr(r0<<1)*sqr_dc-sqr(alpha_hat);
      mpz_set(result[0],sqr_beta_hat*sqr(dc.x));
      mpz_set(result[1],sqr_beta_hat*sqr(dc.y));
      mpz_set(result[2],sqr(two_sqr_dc));
    }};
    #undef MOST
    // const decltype(perturbed(circle0)) X[2] = {perturbed(circle0),perturbed(circle1)};
    const auto X = vec(perturbed(circle0), perturbed(circle1));
    exact::Vec2 fr,fs;
    perturbed_ratio(asarray(fr),&FR::eval,3,asarray(X));
    perturbed_ratio(asarray(fs),&FS::eval,6,asarray(X),true);
    fs = rotate_left_90(fs*exact::Vec2(axis_less<0>(X[0],X[1])?1:-1,
                                       axis_less<1>(X[0],X[1])?1:-1));
#if CHECK
    GEODE_ASSERT(   check_linear.x.thickened(.5).contains(fr.x)
                 && check_linear.y.thickened(.5).contains(fr.y));
    GEODE_ASSERT(   check_quadratic.x.thickened(.5).contains(fs.x)
                 && check_quadratic.y.thickened(.5).contains(fs.y));
#endif

    #if GEODE_KEEP_INTERSECTION_INTERVALS
      v.x._approx_interval = approx_interval(fr - fs);
      v.y._approx_interval = approx_interval(fr + fs);
    #else
      v.x._rounded = fr - fs;
      v.y._rounded = fr + fs;
    #endif

  }

  return v;
}

// Get the canonical intersections for an order pair of circles
template<Pb PS> static ApproxIntersection circle_circle_approx_intersection(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr) {
  return circle_circle_approx_intersections<PS>(cl, cr).x;
}


template<Pb PS> Vector<CircleIntersection<PS>,2> get_intersections(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1) {
  assert(has_intersections(c0,c1));
  const auto approx = circle_circle_approx_intersections<PS>(c0, c1); // Use shared terms to get approx intersection
  return vec(CircleIntersection<PS>(c0,c1,approx.x), CircleIntersection<PS>(c1,c0,approx.y));
}

template<Pb PS> Vector<HorizontalIntersection<PS>,2> get_intersections(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1) {
  assert(has_intersections(c0,h1));
  const auto as_incident = c0.get_intersections(h1);
  return vec(HorizontalIntersection<PS>(as_incident.x, c0), HorizontalIntersection<PS>(as_incident.y, c0));
}



template<Pb PS> SmallArray<CircleIntersection<PS>,2> intersections_if_any(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1) {
  return !is_same_circle(c0, c1) && has_intersections(c0,c1)
    ? SmallArray<CircleIntersection<PS>,2>(get_intersections(c0, c1))
    : SmallArray<CircleIntersection<PS>,2>();
}

template<Pb PS> SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1) {
  return has_intersections(c0,h1) ? SmallArray<HorizontalIntersection<PS>,2>(get_intersections(c0,h1))
                                  : SmallArray<HorizontalIntersection<PS>,2>();
}

template<Pb PS> SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const ExactArc<PS>& a0, const ExactHorizontal<PS>& h1) {
  SmallArray<HorizontalIntersection<PS>, 2> result;
  for(const auto i : a0.circle.intersections_if_any(h1)) {
    if(a0.contains_horizontal(i))
      result.append(HorizontalIntersection<PS>(i, a0.circle));
  }
  return result;
}

template<Pb PS> SmallArray<CircleIntersection<PS>,2> intersections_if_any(const ExactArc<PS>& a0, const ExactArc<PS>& a1) {
  SmallArray<CircleIntersection<PS>, 2> result;
  for(const auto i : intersections_if_any(a0.circle, a1.circle)) {
    if(a0.interior_contains(i) && a1.interior_contains(i))
      result.append(i);
  }
  return result;
}

namespace {
struct RightwardsA { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1, const TV S2) {
  return S1.x-S0.x;
}};
template<int i> struct RightwardsC { template<class TV> static PredicateType<2,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto S = choice<i>(S0,S1);
  return sqr(S.z)-sqr(S2.y-S.y);
}};}
template<Pb PS> bool intersections_rightwards(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1) {
  assert(is_same_horizontal(i0.line, i1.line));
  if (is_same_circle(i0.circle, i1.circle)) {
    assert(i0.left != i1.left);
    return i0.left;
  }
  return FILTER(i1.x-i0.x,
                perturbed_predicate_two_sqrts<RightwardsA,One,One,RightwardsC<0>,RightwardsC<1>>(i0.left?1:-1,i1.left?-1:1,perturbed(i0.circle),perturbed(i1.circle),perturbed(i0.line)));
}

template<> ExactCircle<Pb::Implicit>::ExactCircle(const Vector<Quantized,2> center, const Quantized radius)
 : ExactCirclePerturbationHelper<Pb::Implicit>()
 , center(center)
 , radius(radius)
{
  assert(exact::is_quantized(center) && exact::is_quantized(radius));
}

template<> ExactCircle<Pb::Explicit>::ExactCircle(const Vector<Quantized,2> center, const Quantized radius, const int index)
 : ExactCirclePerturbationHelper<Pb::Explicit>(index)
 , center(center)
 , radius(radius)
{ }

template<Pb PS> Vector<IncidentCircle<PS>,2>     ExactCircle<PS>::get_intersections(const ExactCircle<PS>& incident) const {
  const auto approx = circle_circle_approx_intersections<PS>(*this, incident);
  return vec(IncidentCircle<PS>(*this, incident, ReferenceSide::cl, approx.x),
             IncidentCircle<PS>(incident, *this, ReferenceSide::cr, approx.y));
}

template<Pb PS> Vector<IncidentHorizontal<PS>,2> ExactCircle<PS>::get_intersections(const ExactHorizontal<PS>& h) const {
  Vector<IncidentHorizontal<PS>,2> i;
  i.x.line = i.y.line = h;
  i.x.left = false;
  i.y.left = true;
  // Compute quadrants
  const bool below = upwards(perturbed_center(h),perturbed_center(*this));
  i.x.q = 3*below;
  i.y.q = 1+below;
  // Compute interval x coordinates
  const auto s = assume_safe_sqrt(sqr(Interval(radius))-sqr(Interval(center.y)-h.y));
  i.x.x = center.x + s;
  i.y.x = center.x - s;
  return i;
}


#ifndef NDEBUG
// Any time we use an IncidentCircle we must only use the correct matching reference circle
// This won't notice intersections at almost the same place/q but with a different reference circle so it can't reliably match up incidents with reference circles
template<Pb PS, class... Args> static void assert_incident_args(const ExactCircle<PS>& c, const Args&... incidents) {
  for(const IncidentCircle<PS>& i : vec(incidents...)) {
#if 0
    // This is very slow even for a debug build, but will catch pretty much all cases
    const auto dup = (i.side == ReferenceSide::cl) ? c.intersection_min(i.as_circle()) : c.intersection_max(i.as_circle()); // This will throw if i doesn't actually intersect c
    assert(i.q == dup.q && i.box().intersects(dup.box()));
#else
    assert(!is_same_circle(c,i.as_circle())); // Lazy check for the most likely error
#endif
  }
}
#else
#define assert_incident_args(...)
#endif

template<Pb PS> IncidentCircle<PS> ExactCircle<PS>::other_intersection(const IncidentCircle<PS>& i) const {
  assert_incident_args(*this, i);
  const auto result = (i.side != ReferenceSide::cl)
                    ? intersection_min(i)
                    : intersection_max(i);
  assert(is_same_circle(result, i));
  assert(result.side != i.side);
  return result;
}

template<Pb PS> SmallArray<IncidentCircle<PS>,2> ExactCircle<PS>::intersections_if_any(const ExactCircle<PS>& incident) const {
  return !is_same_circle(*this, incident) && has_intersections(*this, incident)
          ? SmallArray<IncidentCircle<PS>,2>(get_intersections(incident))
          : SmallArray<IncidentCircle<PS>,2>();
}

template<Pb PS> SmallArray<IncidentHorizontal<PS>,2> ExactCircle<PS>::intersections_if_any(const ExactHorizontal<PS>& h) const {
  return has_intersections(*this, h)
          ? SmallArray<IncidentHorizontal<PS>,2>(get_intersections(h))
          : SmallArray<IncidentHorizontal<PS>,2>();
}

template<Pb PS> bool ExactCircle<PS>::is_same_intersection(const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const {
  assert_incident_args(*this, i0, i1);
  // Check if two IncidentCircle refer to the same symbolic object
  // Even though '*this' isn't read, this is a method of ExactCircle for consistency with ther IncidentCircle predicates
  const bool result = (i0.side == i1.side) && is_same_circle(i0, i1);
  assert(!result || ((i0.q == i1.q) && (i0.approx.box().intersects(i1.approx.box())))); // if equal, ought to have same approximate data
  return result;
}

template<Pb PS> bool ExactCircle<PS>::intersections_upwards_same_q(const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const {
  assert_incident_args(*this, i0, i1);
  assert(i0.q == i1.q);
  assert(!is_same_intersection(i0, i1));
  assert(!is_same_circle(*this, i0) && !is_same_circle(*this, i1));

  // Quadrants are monotone so we can check x coordinates of intervals in addition to y coordinates for additional pruning
  const int s = weak_sign(i1.p().x-i0.p().x);
  if (s) {
    const bool result = !(i0.q & 1) ^ (s > 0); // Flip if x and y change in different directions
    #if CHECK
    assert(result == circle_intersections_upwards_degenerate(*this, i0, i1));
    #endif
    return result;
  }

  return FILTER(i1.p().y-i0.p().y, circle_intersections_upwards_degenerate(*this, i0, i1));
}

// For use in intersections_upwards
template<Pb PS> static inline bool above_center(const IncidentCircle<PS>& i) { return i.q <= 1; }

template<Pb PS> bool ExactCircle<PS>::intersections_upwards(const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const {
  assert_incident_args(*this, i0, i1);
  if(i0.q == i1.q) // Fall back to same_q version when possible
    return intersections_upwards_same_q(i0, i1);
  #if 0
  if(above_center(i0) != above_center(i1)) { // If center is between intersections, use that to seperate them
    const bool result = above_center(i1);
    #if CHECK
    assert(result == circle_intersections_upwards_degenerate(*this, i0, i1));
    #endif
    return result;
  }
  #endif

  return FILTER(i1.p().y-i0.p().y, circle_intersections_upwards_degenerate(*this, i0, i1));
}

template<Pb PS> static inline bool circle_intersections_sum_upwards_opp_q_degenerate(const ExactCircle<PS>& c0, const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) {
  if(is_same_circle(i0, i1)) {
    return upwards(perturbed_center(c0),perturbed_center(i0)) == perturbed_predicate<Alpha>(perturbed(c0),perturbed(i0));
  }
  else {
    return perturbed_upwards<true>(cl_is_incident(i0.side)?1:-1,cl_is_incident(i1.side)?1:-1,perturbed(c0),perturbed(i0),perturbed(i1));
  }
}

// Test if angle is less than pi by looking at average direction from center of circle to intersections
template<Pb PS> static inline bool circle_intersections_sum_upwards_opp_q(const ExactCircle<PS>& c0, const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) {
  assert(((i0.q + 2) & 3) == i1.q); // Should only be called for intersections in opposite quadrants
  return FILTER(i0.p().y+i1.p().y-2*c0.center.y, circle_intersections_sum_upwards_opp_q_degenerate(c0, i0, i1));
}

template<Pb PS> static inline bool circle_intersections_ccw_degenerate(const ExactCircle<PS>& c0, const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) {
  // Perform case analysis based on the two quadrants
  switch ((i1.q-i0.q)&3) {
    case 0: return c0.intersections_upwards_same_q   (i0,i1) ^ (i0.q==1 || i0.q==2);
    case 2: return circle_intersections_sum_upwards_opp_q(c0,i0,i1) ^ (i0.q==1 || i0.q==2);
    case 3: return false;
    default: return true; // case 1
  }
}

// Are the intersections of a common circle with two others counterclockwise? In other words, is the triangle c0,i0,i1 positively oriented?
template<Pb PS> bool ExactCircle<PS>::intersections_ccw(const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const {
  assert_incident_args(*this, i0, i1);
  const auto center = Vector<Interval,2>(this->center);
  return FILTER(cross(i0.p()-center,i1.p()-center),
                circle_intersections_ccw_degenerate(*this,i0,i1));
}

template<Pb PS> bool ExactCircle<PS>::intersections_ccw_same_q(const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const {
  assert_incident_args(*this, i0, i1);
  assert(i0.q == i1.q);
  const auto q = i0.q;
  return intersections_upwards_same_q(i0, i1) ^ ( (q == 1) || (q == 2) );
}

template<Pb PS> bool ExactCircle<PS>::intersections_ccw_same_q(const IncidentCircle<PS>& i0, const IncidentHorizontal<PS>& i1) const {
  assert_incident_args(*this, i0);
  assert(i0.q == i1.q);
  const auto q = i0.q;
  return intersections_upwards(i0, i1) ^ ( (q == 1) || (q == 2) );
}

namespace {
struct HorizontalA { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  const auto y = S2.y;
  const auto dc = c1-c0;
  const auto sqr_dc = esqr_magnitude(dc);
  return (((y-c0.y)<<1)-dc.y)*sqr_dc-dc.y*(r0+r1)*(r0-r1);
}};
struct HorizontalB { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1, const TV S2) {
  return S0.x-S1.x;
}};}
template<Pb PS> bool ExactCircle<PS>::intersections_upwards(const IncidentCircle<PS>& i, const IncidentHorizontal<PS>& h) const {
  assert_incident_args(*this, i);
  return FILTER(h.line.y-i.p().y,
                perturbed_predicate_sqrt<HorizontalA,HorizontalB,Beta<0,1>>(cl_is_incident(i.side)?1:-1,perturbed(*this),perturbed(i),perturbed(h.line)));
}

static Interval approx_angle_helper(const Vector<Quantized,2> center, const Vector<Interval,2> approx, const uint8_t q) {
  assert(0 <= q && q < 4);
  auto delta = rotate_left_90_times(approx - Vector<Interval,2>(center), -q); // Rotate vector into first quadrant
  delta = Vector<Interval,2>::componentwise_max(delta, vec(Interval(0),Interval(0))); // Clamp value to first quadrant

  if(!delta.x.contains_zero()) {
    const Interval pos_theta = atan(delta.y * inverse(delta.x));
    return (Interval(half_pi)*(q+0)) + pos_theta;
  }
  if(!delta.y.contains_zero()) {
    const Interval neg_theta = atan(delta.x * inverse(delta.y));
    return (Interval(half_pi)*(q+1)) - neg_theta;
  }
  return Interval(half_pi)*q + Interval(0, half_pi);
}

template<Pb PS> Interval ExactCircle<PS>::approx_angle(const IncidentCircle<PS>& i) const {
  assert_incident_args(*this, i);
  return approx_angle_helper(center, i.approx.p(), i.q);
}
template<Pb PS> Interval ExactCircle<PS>::approx_angle(const IncidentHorizontal<PS>& i) const {
  return approx_angle_helper(center, i.p(), i.q);
}


template<Pb PS> IncidentCircle<PS>::IncidentCircle(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ReferenceSide _side)
 : ExactCircle<PS>(cl_is_incident(_side) ? cl : cr)
 , approx(circle_circle_approx_intersection<PS>(cl, cr))
 , q(circle_circle_intersection_quadrant(_side, cl, cr, approx))
 , side(_side)
{ }

template<Pb PS> IncidentCircle<PS>::IncidentCircle(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ReferenceSide _side, const ApproxIntersection _approx)
 : ExactCircle<PS>(cl_is_incident(_side) ? cl : cr)
 , approx(_approx)
 , q(circle_circle_intersection_quadrant(_side, cl, cr, approx))
 , side(_side)
{ }

template<Pb PS> IncidentCircle<PS> IncidentCircle<PS>::reference_as_incident(const ExactCircle<PS>& reference) const {
  const ExactCircle<PS>& incident = *this;
  assert(!is_same_circle(reference, incident));
  assert(has_intersections(reference, incident));
  assert_incident_args(reference, *this);
  const ExactCircle<PS>& cl = cl_is_incident(side) ? incident : reference;
  const ExactCircle<PS>& cr = cl_is_incident(side) ? reference : incident;
  return IncidentCircle(cl, cr, opposite(side), approx);
}

template<Pb PS> CircleIntersectionKey<PS>::CircleIntersectionKey(const ExactCircle<PS>& _cl, const ExactCircle<PS>& _cr)
 : cl(_cl)
 , cr(_cr)
{ }

template<Pb PS> CircleIntersectionKey<PS>::CircleIntersectionKey(const ExactCircle<PS>& reference, const IncidentCircle<PS>& incident)
 : cl(cl_is_reference(incident.side) ? reference : incident)
 , cr(cr_is_reference(incident.side) ? reference : incident)
{ }

template<Pb PS> CircleIntersection<PS>::CircleIntersection(const CircleIntersectionKey<PS>& k, const ApproxIntersection _approx, const uint8_t _ql, const uint8_t _qr)
 : CircleIntersectionKey<PS>(k)
 , approx(_approx)
 , ql(_ql)
 , qr(_qr)
{ }

template<Pb PS> CircleIntersection<PS>::CircleIntersection(const CircleIntersectionKey<PS>& k, const ApproxIntersection _approx)
 : CircleIntersectionKey<PS>(k)
 , approx(_approx)
 , ql(circle_circle_intersection_quadrant(ReferenceSide::cl, this->cl, this->cr, approx))
 , qr(circle_circle_intersection_quadrant(ReferenceSide::cr, this->cl, this->cr, approx))
{ }

template<Pb PS> CircleIntersection<PS>::CircleIntersection(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ApproxIntersection _approx)
 : CircleIntersection(CircleIntersectionKey<PS>(cl,cr), _approx)
{ }


template<Pb PS> CircleIntersection<PS>::CircleIntersection(const ExactCircle<PS>& reference, const IncidentCircle<PS>& incident)
 : CircleIntersectionKey<PS>(reference, incident)
 , approx(incident.approx)
 , ql(cl_is_reference(incident.side) ? incident.q : circle_circle_intersection_quadrant(ReferenceSide::cl, this->cl, this->cr, approx))
 , qr(cr_is_reference(incident.side) ? incident.q : circle_circle_intersection_quadrant(ReferenceSide::cr, this->cl, this->cr, approx))
{
  // Check that copied value from incident.q was correct
  assert(ql == circle_circle_intersection_quadrant(ReferenceSide::cl, this->cl, this->cr, approx));
  assert(qr == circle_circle_intersection_quadrant(ReferenceSide::cr, this->cl, this->cr, approx));
}

template<Pb PS> CircleIntersection<PS>::CircleIntersection(const CircleIntersectionKey<PS>& k)
 : CircleIntersectionKey<PS>(k)
 , approx(circle_circle_approx_intersection(this->cl, this->cr))
 , ql(circle_circle_intersection_quadrant(ReferenceSide::cl, this->cl, this->cr, approx))
 , qr(circle_circle_intersection_quadrant(ReferenceSide::cr, this->cl, this->cr, approx))
{ }



// Is the (a0,a1) intersection inside circle b?  Degree 8
namespace {
// This predicate has the form
//   A + B sqrt(Beta<0,1>) < 0
// where
struct CircleIntersectionInsideCircle_A { template<class TV> static PredicateType<4,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
  const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z;
  const auto c01 = c1-c0, c02 = c2-c0;
  const auto sqr_c01 = esqr_magnitude(c01),
             sqr_c02 = esqr_magnitude(c02),
             alpha01 = sqr_c01+(r0+r1)*(r0-r1),
             alpha02 = sqr_c02+(r0+r2)*(r0-r2);
  return alpha01*edot(c01,c02)-alpha02*sqr_c01;
}};
struct CircleIntersectionInsideCircle_B { template<class TV> static PredicateType<2,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
  return edet(c1-c0,c2-c0);
}};
}

template<Pb PS> bool CircleIntersection<PS>::is_inside(const ExactCircle<PS>& c) const {
  return FILTER(sqr(Interval(c.radius))-sqr_magnitude(this->approx.p()-Vector<Interval,2>(c.center)),
                perturbed_predicate_sqrt<CircleIntersectionInsideCircle_A,CircleIntersectionInsideCircle_B,Beta<0,1>>(-1,perturbed(this->cl),perturbed(this->cr),perturbed(c)));
}

template<Pb PS> bool ExactArc<PS>::is_full_circle() const {
  return circle.is_same_intersection(src, dst);
}

template<Pb PS> Vec2 ExactArc<PS>::q_and_opp_q() const {
  assert(!is_full_circle()); // q isn't defined for a full circle

  const auto r = circle.radius;
  const auto x0 = src.approx.guess();
  const auto x1 = dst.approx.guess();
  const auto l = min(r, 0.5 * (x0 - x1).magnitude());

  const auto xm = 0.5*(x0+x1); // Compute the midpoint of the chord from x0 to x1
  const auto h = min(r,(circle.center - xm).magnitude()); // Height from chord to center of circle (clamped in case error in xm moved it outside of circle)
  // There are many different ways we could compute q, of which I considered several in depth:
  //   q = l / (r + sqrt(sqr(r) - sqr(l)))  // (argument to sqrt needs to be clamped to zero to handle rounding errors)
  //   q = sqrt((r-h) / (r+h))
  //   q = l / (r + h)
  // We have to be careful about noise in x0 and x1 as well as floating point stability
  // Since endpoints are fixed regardless of q, we want to compute a value that will match the intended arc in the middle
  // The first approach fits a fixed radius to inaccurate endpoints making it unstable when q is close to +/-1 resulting in large worst case errors
  // By including information from center of circle (which is exact) we should get more stable position of arc midpoint
  // I choose the third approach over the second since it ensures q is 0 when l is 0
  const real abs_q_short = l / (r+h);
  const real abs_q_long = (r+h) / l;

  const bool is_small_arc = circle.intersections_ccw(src, dst);
  const real q =      (is_small_arc ? abs_q_short : abs_q_long);
  const real opp_q = -(is_small_arc ? abs_q_long : abs_q_short);

  assert(max(abs(q), abs(opp_q)) >= 1 && min(abs(q),abs(opp_q)) <= 1);

  return Vec2(q, opp_q);
}

template<Pb PS> bool ExactArc<PS>::unsafe_contains(const IncidentCircle<PS>& i) const {
  if (src.q != dst.q) { // arc starts and ends in different quadrants
    if (src.q == i.q)
      return circle.intersections_ccw_same_q(src, i);
    else if (dst.q == i.q)
      return circle.intersections_ccw_same_q(i, dst);
    else
      return (((i.q-src.q)&3)<((dst.q-src.q)&3));
  } else { // arc starts and ends in the same quadrant
    if(is_full_circle())
      return true;
    const bool small = circle.intersections_ccw_same_q(src, dst);
    return small ^ (   src.q != i.q
                     || (small ^ circle.intersections_ccw_same_q(src,i))
                     || (small ^ circle.intersections_ccw_same_q(i,dst)));
  }
}

template<Pb PS> bool ExactArc<PS>::interior_contains(const IncidentCircle<PS>& i) const {
  return has_endpoint(i) ? is_full_circle() : unsafe_contains(i);
}
template<Pb PS> bool ExactArc<PS>::half_open_contains(const IncidentCircle<PS>& i) const {
  if(circle.is_same_intersection(src,i))
    return true; // Include start
  if(circle.is_same_intersection(dst,i)) {
    assert(!is_full_circle()); // Should only be a full circle if src == dst in which case we shouldn't have dst == i since we just checked src == i
    return false;
  }
  return unsafe_contains(i); // Fall back to general case
}

template<Pb PS> Box<exact::Vec2> bounding_box(const ExactArc<PS>& a) {
  // We start with the bounding box of the endpoints
  auto box = Box<exact::Vec2>::combine(a.src.box(),a.dst.box());

  if(a.src.q == a.dst.q) {
    // If src and dst are in same quadrant, arc will either hit all 4 axis or none
    if(a.is_full_circle() || !a.circle.intersections_ccw_same_q(a.src, a.dst))
      return bounding_box(a.circle);
    else
      return box;
  }
  else {
    // If src and dst are in different quadrants we update each crossed axis
    auto q = a.src.q;
    do {
      q = (q+1)&3; // Step to next quadrant
      switch(q) {
        // Add start of new quadrant
        // Arc bounding box must be a subset of the circle bounding box so we can directly update each axis as we cross into the quadrant
        case 0: box.max.x = a.circle.center.x + a.circle.radius; break;
        case 1: box.max.y = a.circle.center.y + a.circle.radius; break;
        case 2: box.min.x = a.circle.center.x - a.circle.radius; break;
        case 3: box.min.y = a.circle.center.y - a.circle.radius; break;
        GEODE_UNREACHABLE();
      }
    } while(q != a.dst.q); // Go until we end up at dst
    return box;
  }
}

template<Pb PS> bool arcs_overlap(const ExactArc<PS>& a0, const ExactArc<PS>& a1) {
  assert(is_same_circle(a0.circle, a1.circle));
  // This duplicates a lot of comparisons. It'd probably be faster to sort all endpoints once but tricky to handle all of the corner cases
  return a0.interior_contains(a1.src)
      || a0.interior_contains(a1.dst)
      || a1.interior_contains(a0.src)
      || a1.interior_contains(a0.src);
}

template<Pb PS> bool ExactArc<PS>::contains_horizontal(const IncidentHorizontal<PS>& h) const {
  if(src.q != dst.q) {
    if(src.q == h.q)
      return circle.intersections_ccw_same_q(src, h);
    else if(dst.q == h.q)
      return circle.intersections_ccw_same_q(h, dst);
    else
      return ((h.q-src.q)&3) < ((dst.q-src.q)&3);
  } else { // arc starts and ends in the same quadrant
    if(is_full_circle())
      return true;
    const bool small = circle.intersections_ccw_same_q(src, dst);
    if(src.q != h.q)
      return !small; // Small arcs don't contain h if it is in another quadrant
    if(small)
      return circle.intersections_ccw_same_q(src, h) && circle.intersections_ccw_same_q(h, dst); // Include everything after src and before dst
    else
      return circle.intersections_ccw_same_q(src, h) || circle.intersections_ccw_same_q(h, dst); // Arc covers from src to end of quadrant and wraps around to cover quadrant up to dst
  }
}

template<Pb PS> bool ExactHorizontalArc<PS>::contains(const IncidentCircle<PS>& o) const {
  const IncidentCircle<PS>& src = i;
  const IncidentHorizontal<PS>& dst = h;
  const bool flipped = h_is_src;

  if (src.q != dst.q) { // arc starts and ends in different quadrants
    if (src.q == o.q)
      return flipped ^ circle.intersections_ccw_same_q(src, o);
    else if (dst.q == o.q)
      return flipped ^ circle.intersections_ccw_same_q(o, dst);
    else
      return flipped ^ (((o.q-src.q)&3)<((dst.q-src.q)&3));
  } else { // arc starts and ends in the same quadrant
    const bool small = circle.intersections_ccw_same_q(src, dst);
    return flipped ^ small ^ (   src.q != o.q
                               || (small ^ circle.intersections_ccw_same_q(src,o))
                               || (small ^ circle.intersections_ccw_same_q(o,dst)));
  }
}

template<Pb PS> Box<exact::Vec2> bounding_box(const ExactHorizontalArc<PS>& a) {
  const IncidentCircle<PS>& src = a.i;
  const IncidentHorizontal<PS>& dst = a.h;
  const bool flipped = a.h_is_src;

  // We start with the bounding box of the endpoints
  auto box = Box<exact::Vec2>::combine(src.box(),dst.box());

  if(src.q == dst.q) {
    // If src and dst are in same quadrant, arc will either hit all 4 axis or none
    if(flipped ^ !a.circle.intersections_ccw_same_q(src, dst))
      return bounding_box(a.circle);
    else
      return box;
  }
  else {
    // If src and dst are in different quadrants we update each crossed axis
          auto q     = a.h_is_src ? a.h.q : a.i.q;
    const auto dst_q = a.h_is_src ? a.i.q : a.h.q;
    do {
      q = (q+1)&3; // Step to next quadrant
      switch(q) {
        // Add start of new quadrant
        // Arc bounding box must be a subset of the circle bounding box so we can directly update each axis as we cross into the quadrant
        case 0: box.max.x = a.circle.center.x + a.circle.radius; break;
        case 1: box.max.y = a.circle.center.y + a.circle.radius; break;
        case 2: box.min.x = a.circle.center.x - a.circle.radius; break;
        case 3: box.min.y = a.circle.center.y - a.circle.radius; break;
        GEODE_UNREACHABLE();
      }
    } while(q != dst_q); // Go until we end up at dst
    return box;
  }
}

#define INSTANTIATE(PS) \
  template struct ExactCircle<PS>; \
  template struct IncidentCircle<PS>; \
  template struct CircleIntersectionKey<PS>; \
  template struct CircleIntersection<PS>; \
  template struct HorizontalIntersection<PS>; \
  template struct ExactArc<PS>; \
  template struct ExactHorizontalArc<PS>; \
  template bool is_same_circle      (const            ExactCircle<PS>& c0, const            ExactCircle<PS>& c1); \
  template bool is_same_horizontal  (const        ExactHorizontal<PS>& h0, const        ExactHorizontal<PS>& h1); \
  template bool is_same_intersection(const  CircleIntersectionKey<PS>& i0, const  CircleIntersectionKey<PS>& i1); \
  template bool is_same_intersection(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1); \
  template bool has_intersections(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1); \
  template bool circles_overlap(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1); \
  template SmallArray<CircleIntersection<PS>,2> intersections_if_any(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1); \
  template SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const ExactCircle<PS>& a0, const ExactHorizontal<PS>& h1); \
  template SmallArray<CircleIntersection<PS>,2> intersections_if_any(const ExactArc<PS>& a0, const ExactArc<PS>& a1); \
  template SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const ExactArc<PS>& a0, const ExactHorizontal<PS>& h1); \
  template bool intersections_rightwards(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1); \
  template bool arcs_overlap(const ExactArc<PS>& a0, const ExactArc<PS>& a1); \
  template Box<exact::Vec2> bounding_box(const ExactCircle<PS>& c); \
  template Box<exact::Vec2> bounding_box(const ExactArc<PS>& a); \
  template Box<exact::Vec2> bounding_box(const ExactHorizontalArc<PS>& a);

INSTANTIATE(Pb::Explicit)
INSTANTIATE(Pb::Implicit)
#undef INSTANTIATE

} // namespace geode
