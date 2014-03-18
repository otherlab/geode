#include <geode/exact/circle_predicates.h>
#include <geode/exact/Exact.h>
#include <geode/exact/Interval.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/exact/predicates.h>
#include <geode/exact/scope.h>
#include <geode/utility/str.h>
namespace geode {
typedef exact::Vec2 EV2;
typedef Vector<Exact<1>,3> LV3;
using std::cout;
using std::endl;


// If true, always run both fast and slow tests and compare results
// IMPORTANT: This is a much stronger test than the pure unit tests, and should be run whenver this file is changed.
#define CHECK 0

// Run a fast interval check, and fall back to a slower exact check if it fails.  If check is true, do both and validate.
#if !CHECK
#ifdef __GNUC__
// In gcc, we can define a clean macro that evaluates its arguments at most once time.
#define FILTER(fast,...) ({ \
  const int _s = weak_sign(fast); \
  _s ? _s>0 : __VA_ARGS__; })
#else
// Warning: we lack gcc, the argument must be evaluated multiple times.  Hopefully CSE will do its work.
#define FILTER(fast,...) \
  (  certainly_positive(fast) ? true \
   : certainly_negative(fast) ? false \
   : __VA_ARGS__)
#endif
#else
// In check mode, always do both.
static bool filter_helper(const Interval fast, const bool slow, const int line) {
  const int sign = weak_sign(fast);
  if (sign && (sign>0)!=slow)
    throw AssertionError(format("circle_csg: Consistency check failed on line %d, interval %s, slow sign %d",line,str(fast),slow?1:-1));
  return slow;
}
#define FILTER(fast,...) filter_helper(fast,__VA_ARGS__,__LINE__)
#endif

static Vertex make_placeholder_vertex(Arcs arcs, int i0) {
  Vertex result;
  result.i0 = i0;
  result.i1 = i0;
  result.left = true;
  result.q0 = result.q1 = 0;
  result.rounded = exact::Vec2::repeat(numeric_limits<Quantized>::quiet_NaN());
  return result;
}

// Check if an arc has the same (symbolic) vertex repeated (should indicate a full circle)
bool arc_is_repeated_vertex(Arcs arcs, const Vertex& v01, const Vertex& v12) {
  assert(v01.i1 == v12.i0);
  return arcs_from_same_circle(arcs, v01.i0, v12.i1) && 
    (v01.left != v12.left || arcs_from_same_circle(arcs, v01.i0, v12.i0));
}

// Do two circles intersect (degree 2)?
namespace {
template<bool add> struct Intersect {
  template<class TV> static inline PredicateType<2,TV> eval(const TV S0, const TV S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return sqr(add?r1+r0:r1-r0)-esqr_magnitude(c1-c0);
}};}
bool circles_intersect(Arcs arcs, const int arc0, const int arc1) {
  return     perturbed_predicate<Intersect<true >>(aspoint(arcs,arc0),aspoint(arcs,arc1))
         && !perturbed_predicate<Intersect<false>>(aspoint(arcs,arc0),aspoint(arcs,arc1));
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
// This function should be used only from circle_circle_intersections, where it is precomputed as Vertex::q0.
// As written this is degree 6, but it can be reduced to degree 2 if necessary.
namespace {
template<int axis> struct QuadrantA { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1) {
  return Alpha::eval(S0,S1)*(axis==0?S1.y-S0.y:S0.x-S1.x);
}};
template<int axis> struct QuadrantB { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1) {
  return S1[axis]-S0[axis]; // dc[axis]
}};}
static int circle_circle_intersection_quadrant(Arcs arcs, const Vertex v) {
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
  const auto center = arcs[v.i0].center;
  const bool p0 = FILTER(v.p().y-center.y, perturbed_predicate_sqrt<QuadrantA<0>,QuadrantB<0>,Beta<0,1>>(v.left?1:-1,aspoint(arcs,v.i0),aspoint(arcs,v.i1))),
             p1 = FILTER(center.x-v.p().x, perturbed_predicate_sqrt<QuadrantA<1>,QuadrantB<1>,Beta<0,1>>(v.left?1:-1,aspoint(arcs,v.i0),aspoint(arcs,v.i1)));
  // Assemble our two predicates into a quadrant
  return 2*!p0+(p0==p1);
}

// Construct both of the intersections of two circular arcs, assuming they do intersect.
// The two intersections are to the right and the left of the center segment, respectively, so result[left] is correct.
// The results differ from the true intersections by at most 2.
// Degrees 3/2 for the nonsqrt part and 6/4 for the part under the sqrt.
Vector<Vertex,2> circle_circle_intersections(Arcs arcs, const int arc0, const int arc1) {
  Vector<Vertex,2> v;
  v.x.i0 = v.y.i0 = arc0;
  v.x.i1 = v.y.i1 = arc1;
  v.x.left = false;
  v.y.left = true;

#if CHECK
  GEODE_WARNING("Expensive consistency checking enabled");
  Vector<Interval,2> check_linear, check_quadratic;
  check_linear.fill(Interval::full());
  check_quadratic.fill(Interval::full());
#endif

  // Evaluate conservatively using intervals
  {
    const Vector<Interval,2> c0(arcs[arc0].center), c1(arcs[arc1].center);
    const Interval           r0(arcs[arc0].radius), r1(arcs[arc1].radius);
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
      if (small(linear,1)) {
        const auto beta_hat = assume_safe_sqrt((sqr_r0*(sqr_dc<<2))-sqr(alpha_hat));
        const auto quadratic = half_inv_sqr_dc*beta_hat*rotate_left_90(dc);
#if CHECK
        check_quadratic = quadratic;
#endif
        if (small(quadratic,1) && !CHECK) {
          const auto sl = snap(linear),
                     sq = snap(quadratic);
          v.x.rounded = sl-sq;
          v.y.rounded = sl+sq;
          goto quadrants;
        }
      }
    }
  }

  {
    // If intervals fail, evaluate and round using symbolic perturbation.  For simplicity, we round the sqrt part
    // separately from the rational part, at the cost of a maximum error of 2.  The full formula is
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
    const exact::Point3 X[2] = {aspoint(arcs,arc0),aspoint(arcs,arc1)};
    exact::Vec2 fr,fs;
    perturbed_ratio(asarray(fr),&FR::eval,3,asarray(X));
    perturbed_ratio(asarray(fs),&FS::eval,6,asarray(X),true);
    fs = rotate_left_90(fs*EV2(axis_less<0>(X[0],X[1])?1:-1,
                               axis_less<1>(X[0],X[1])?1:-1));
#if CHECK
    GEODE_ASSERT(   check_linear.x.thickened(1).contains(fr.x)
                 && check_linear.y.thickened(1).contains(fr.y));
    GEODE_ASSERT(   check_quadratic.x.thickened(1).contains(fs.x)
                 && check_quadratic.y.thickened(1).contains(fs.y));
#endif
    v.x.rounded = fr - fs;
    v.y.rounded = fr + fs;
  }

  // Fill in quadrants
quadrants:
  v.x.q0 = circle_circle_intersection_quadrant(arcs,v.x);
  v.x.q1 = circle_circle_intersection_quadrant(arcs,v.x.reverse());
  v.y.q0 = circle_circle_intersection_quadrant(arcs,v.y);
  v.y.q1 = circle_circle_intersection_quadrant(arcs,v.y.reverse());
  return v;
}

Box<exact::Vec2> arc_box(Arcs arcs, const Vertex& v01, const Vertex& v12) {
  assert(v01.i1 == v12.i0);
  const int i1 = v01.i1;
  if(arc_is_repeated_vertex(arcs, v01, v12)) {
    return Box<EV2>(arcs[i1].center).thickened(arcs[i1].radius);
  }
  // Probably not worth accounting for, but vertex.box() must be inside Box<EV2>(arcs[i].center).thickened(arcs[i].radius) for i in [vertex.i1,vertex.i2]
  auto box = bounding_box(v01.rounded,v12.rounded).thickened(Vertex::tolerance());
  int q0 = v01.q1,
      q1 = v12.q0;
  if (q0==q1) {
    if (arcs[i1].positive != (circle_intersections_upwards(arcs,v01.reverse(),v12) ^ (v01.q1==1 || v01.q1==2))) {
      // The arc hits all four axes
      box.enlarge(Box<EV2>(arcs[i1].center).thickened(arcs[i1].radius));
    } // The arc stays within one quadrant, the endpoints suffice for the bounding box
  } else {
    // The arc hits some axes but not others.  Loop around making the necessary bounding box enlargements.
    if (!arcs[i1].positive)
      swap(q0,q1);
    const auto a1 = arcs[i1];
    while (q0 != q1) {
      box.enlarge(a1.center+rotate_left_90_times(EV2(a1.radius,0),q0+1));
      q0 = (q0+1)&3;
    }
  }
  return box;
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

template<bool add> bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b) {
  assert(a!=b && a.i0==b.i0);
  return FILTER(add ? a.p().y+b.p().y-2*arcs[a.i0].center.y : b.p().y-a.p().y,
               (   arcs_from_same_circle(arcs,a.i1,b.i1)
                || (arcs_from_same_circle(arcs,a.i0,b.i1) && arcs_from_same_circle(arcs,a.i1,a.i0)))
                  ? add ?    upwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) == perturbed_predicate<Alpha>(aspoint(arcs,a.i0),aspoint(arcs,a.i1))
                        : rightwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) ^ a.left
                  : perturbed_upwards<add>(a.left^add?-1:1,b.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b.i1)));
}

// Are the intersections of two circles with a third counterclockwise?  In other words, is the triangle c0,x01,x02 positively oriented?
// The two intersections are assumed to exist.
static bool circle_intersections_ccw_helper(Arcs arcs, const Vertex v0, const Vertex v1) {
  assert(v0.i0==v1.i0);
  // Perform case analysis based on the two quadrants
  const int q0 = v0.q0,
            q1 = v1.q0;
  switch ((q1-q0)&3) {
    case 0: return circle_intersections_upwards      (arcs,v0,v1) ^ (q0==1 || q0==2);
    case 2: return circle_intersections_upwards<true>(arcs,v0,v1) ^ (q0==1 || q0==2);
    case 3: return false;
    default: return true; // case 1
  }
}
// Tests if an arc segment is less then a half circle
bool circle_intersections_ccw(Arcs arcs, const Vertex v01, const Vertex v02) {
  assert(v01.i0==v02.i0);
  const Vector<Interval,2> center(arcs[v01.i0].center);
  return FILTER(cross(v01.p()-center,v02.p()-center),
                circle_intersections_ccw_helper(arcs,v01,v02));
}

// Does the (a1,b) intersection occur on the piece of a1 between a0 and a2?  a1 and b are assumed to intersect.
bool circle_arc_intersects_circle(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a1b) {
  assert(a01.i1==a12.i0 && a12.i0==a1b.i0);
  if(arc_is_repeated_vertex(arcs, a01, a12))
    return true; // Repeated vertex indicates arc covers full circle (If the assumed intersection exists it is part of the arc)
  const auto a10 = a01.reverse();
  const bool flip = !arcs[a01.i1].positive;
  const int q0 = a01.q1,
            q2 = a12.q0,
            qb = a1b.q0;
  const bool qb_down = qb==1 || qb==2;
  if (q0!=q2) { // a012 starts and ends in different quadrants
    if (q0==qb)
      return flip ^ qb_down ^ circle_intersections_upwards(arcs,a10,a1b);
    else if (q2==qb)
      return flip ^ qb_down ^ circle_intersections_upwards(arcs,a1b,a12);
    else
      return flip ^ (((qb-q0)&3)<((q2-q0)&3));
  } else { // a012 starts and ends in the same quadrant
    const bool small = circle_intersections_upwards(arcs,a10,a12) ^ (q0==1 || q0==2);
    return flip ^ small ^ (   q0!=qb
                           || (small ^ qb_down ^ circle_intersections_upwards(arcs,a10,a1b))
                           || (small ^ qb_down ^ circle_intersections_upwards(arcs,a1b,a12)));
  }
}

// Does the piece of a1 between a0 and a1 intersect the piece of b1 between b0 and b2?  a1 and b1 are assumed to intersect.
bool circle_arcs_intersect(Arcs arcs, const Vertex a01, const Vertex a12,
                                      const Vertex b01, const Vertex b12,
                                      const Vertex ab) {
  return circle_arc_intersects_circle(arcs,a01,a12,ab)
      && circle_arc_intersects_circle(arcs,b01,b12,ab.reverse());
}

// Is the (a0,a1) intersection inside circle b?  Degree 8, but can be eliminated entirely.
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
bool circle_intersection_inside_circle(Arcs arcs, const Vertex a, const int b) {
  return FILTER(sqr(Interval(arcs[b].radius))-sqr_magnitude(a.p()-Vector<Interval,2>(arcs[b].center)),
                perturbed_predicate_sqrt<CircleIntersectionInsideCircle_A,CircleIntersectionInsideCircle_B,Beta<0,1>>(a.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b)));
}

// Is the (a0,a1) intersection to the right of b's center?  Degree 6, but can be eliminated entirely.
namespace {
// This predicate has the form
//   (\hat{alpha} c01.x - 2 c02.x c01^2) - c01.y sqrt(Beta<0,1>)
struct CircleIntersectionRightOfCenter_A { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  const auto c01 = c1-c0;
  const auto sqr_c01 = esqr_magnitude(c01),
             alpha = sqr_c01+(r0+r1)*(r0-r1);
  return alpha*c01.x-((c2.x-c0.x)<<1)*sqr_c01;
}};
struct CircleIntersectionRightOfCenter_B { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1, const TV S2) {
  return S1.y-S0.y;
}};
}
bool circle_intersection_right_of_center(Arcs arcs, const Vertex a, const int b) {
  return FILTER(a.p().x-arcs[b].center.x,
                perturbed_predicate_sqrt<CircleIntersectionRightOfCenter_A,CircleIntersectionRightOfCenter_B,Beta<0,1>>(a.left?-1:1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b)));
}

Array<Vertex> compute_vertices(Arcs arcs, RawArray<const int> next) {
  IntervalScope scope;
  Array<Vertex> vertices(arcs.size(),false); // vertices[i] is the start of arcs[i]
  for (int i0=0;i0<arcs.size();i0++) {
    const int i1 = next[i0];
    if(i0 == i1) {
      vertices[i1] = make_placeholder_vertex(arcs, i0);
    }
    else {
      vertices[i1] = circle_circle_intersections(arcs,i0,i1)[arcs[i0].left];
    }
  }
  return vertices;
}

template<int d=3> static inline typename exact::Point<d>::type aspoint_horizontal(const Quantized y) {
  const int index = numeric_limits<int>::max();
  return tuple(index,Vector<Quantized,d>(vec(0,y)));
}

namespace {
template<int sign> struct CircleIntersectsHorizontal { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1) {
  const auto cy = S0.y,
             r = S0.z,
             y = S1.y;
  return sign>0 ? (cy+r)-y
                : y-(cy-r);
}};}
bool circle_intersects_horizontal(Arcs arcs, const int arc, const Quantized y) {
  const auto& a = arcs[arc];
  return FILTER((Interval(a.center.y)+a.radius)-y,
                perturbed_predicate<CircleIntersectsHorizontal<+1>>(aspoint(arcs,arc),aspoint_horizontal(y)))
      && FILTER(y-(Interval(a.center.y)-a.radius),
                perturbed_predicate<CircleIntersectsHorizontal<-1>>(aspoint(arcs,arc),aspoint_horizontal(y)));
}

Vector<HorizontalVertex,2> circle_horizontal_intersections(Arcs arcs, const int arc, const Quantized y) {
  Vector<HorizontalVertex,2> i;
  i.x.arc = i.y.arc = arc;
  i.x.y = i.y.y = y;
  i.x.left = false;
  i.y.left = true;
  // Compute quadrants
  const bool below = upwards(aspoint_horizontal<2>(y),aspoint_center(arcs,arc));
  i.x.q0 = 3*below;
  i.y.q0 = 1+below;
  // Compute interval x coordinates
  const auto a = arcs[arc];
  const auto s = assume_safe_sqrt(sqr(Interval(a.radius))-sqr(Interval(a.center.y)-y));
  i.x.x = a.center.x + s;
  i.y.x = a.center.x - s;
  return i;
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
bool circle_intersection_below_horizontal(Arcs arcs, const Vertex a01, const HorizontalVertex a0y) {
  assert(a01.i0==a0y.arc);
  return FILTER(a0y.y-a01.p().y,
                perturbed_predicate_sqrt<HorizontalA,HorizontalB,Beta<0,1>>(a01.left?1:-1,aspoint(arcs,a01.i0),aspoint(arcs,a01.i1),aspoint_horizontal(a0y.y)));
}

bool circle_arc_contains_horizontal_intersection(Arcs arcs, const Vertex a01, const Vertex a12, const HorizontalVertex a1y) {
  assert(a01.i1==a12.i0 && a12.i0==a1y.arc);
  const auto a10 = a01.reverse();
  const bool flip = !arcs[a01.i1].positive;
  const int q0 = a01.q1,
            q2 = a12.q0,
            qy = a1y.q0;
  const bool qy_down = qy==1 || qy==2;
  if (q0!=q2) { // a012 starts and ends in different quadrants
    if (q0==qy)
      return flip ^ qy_down ^  circle_intersection_below_horizontal(arcs,a10,a1y);
    else if (q2==qy)
      return flip ^ qy_down ^ !circle_intersection_below_horizontal(arcs,a12,a1y);
    else
      return flip ^ (((qy-q0)&3)<((q2-q0)&3));
  } else { // a012 starts and ends in the same quadrant
    if(arc_is_repeated_vertex(arcs,a01,a12))
      return true;
    const bool small = circle_intersections_upwards(arcs,a10,a12) ^ (q0==1 || q0==2);
    return flip ^ small ^ (   q0!=qy
                           || (small ^ qy_down ^  circle_intersection_below_horizontal(arcs,a10,a1y))
                           || (small ^ qy_down ^ !circle_intersection_below_horizontal(arcs,a12,a1y)));
  }
}

namespace {
struct RightwardsA { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1, const TV S2) {
  return S1.x-S0.x;
}};
template<int i> struct RightwardsC { template<class TV> static PredicateType<2,TV> eval(const TV S0, const TV S1, const TV S2) {
  const auto S = choice<i>(S0,S1);
  return sqr(S.z)-sqr(S2.y-S.y);
}};}
bool horizontal_intersections_rightwards(Arcs arcs, const HorizontalVertex ay, const HorizontalVertex by) {
  assert(ay!=by && ay.y==by.y);
  if (arcs_from_same_circle(arcs, ay.arc, by.arc)) {
    assert(ay.left != by.left);
    return ay.left;
  }
  return FILTER(by.x-ay.x,
                perturbed_predicate_two_sqrts<RightwardsA,One,One,RightwardsC<0>,RightwardsC<1>>(ay.left?1:-1,by.left?-1:1,aspoint(arcs,ay.arc),aspoint(arcs,by.arc),aspoint_horizontal(ay.y)));
}

} // namespace geode
