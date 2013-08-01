#include <other/core/exact/circle_predicates.h>
#include <other/core/exact/Exact.h>
#include <other/core/exact/Interval.h>
#include <other/core/exact/math.h>
#include <other/core/exact/perturb.h>
#include <other/core/exact/predicates.h>
#include <other/core/exact/scope.h>
#include <other/core/utility/str.h>
namespace other {
typedef exact::Vec2 EV2;
typedef Vector<Exact<1>,3> LV3;


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


bool arcs_from_same_circle(Arcs arcs, int i0, int i1) {
  if (i0 == i1)
    return true;
  const auto &a0 = arcs[i0],
             &a1 = arcs[i1];
  if (a0.index != a1.index)
    return false;
  assert(a0.center == a1.center && a0.radius == a1.radius && a0.positive == a1.positive);
  return true;
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
// As written this is degree 4, but it can be reduced to degree 2 if necessary.
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
  OTHER_WARNING("Expensive consistency checking enabled");
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
    OTHER_ASSERT(   check_linear.x.thickened(1).contains(fr.x)
                 && check_linear.y.thickened(1).contains(fr.y));
    OTHER_ASSERT(   check_quadratic.x.thickened(1).contains(fs.x)
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
  const int i1 = v01.i1;
  assert(v01.i1 == v12.i0);

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


// Is intersection (a0,a1).y < (b0,b1).y?  This is degree 20 as written, but can be reduced to 6.
// If add = true, assume a0=b0 and check whether ((0,a1)+(0,b1)).y > 0.
namespace {
template<bool add> struct UpwardsA { template<class TV> static PredicateType<5,TV> eval(const TV S0, const TV S1, const TV S2, const TV S3) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy(), c3 = S3.xy();
  const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z,    r3 = S3.z;
  const auto c01 = c1-c0, c23 = c3-c2;
  const auto sqr_c01 = esqr_magnitude(c01),
             sqr_c23 = esqr_magnitude(c23);
  const auto alpha01 = sqr_c01+(r0+r1)*(r0-r1),
             alpha23 = sqr_c23+(r2+r3)*(r2-r3);
  return !add ? ((c2.y-c0.y)<<1)*sqr_c01*sqr_c23+alpha23*(c23.y*sqr_c01)-alpha01*(c01.y*sqr_c23)
              : alpha23*(c23.y*sqr_c01)+alpha01*(c01.y*sqr_c23);
}};
template<int i> struct UpwardsB { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1, const TV S2, const TV S3) {
  BOOST_STATIC_ASSERT(i==0 || i==2);
  const auto c01 = S1.xy()-S0.xy(),
             c23 = S3.xy()-S2.xy();
  return i==0 ? c01.x*esqr_magnitude(c23) // Negated below
              : c23.x*esqr_magnitude(c01);
}};}
template<bool add> bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b) {
  assert(a!=b && a.i0==b.i0);
  return FILTER(add ? a.p().y+b.p().y-(arcs[a.i0].center.y*2) : b.p().y-a.p().y,
               (    (arcs_from_same_circle(arcs, a.i0, b.i0) && arcs_from_same_circle(arcs, a.i1, b.i1))
                 || (arcs_from_same_circle(arcs, a.i0, b.i1) && arcs_from_same_circle(arcs, a.i1, b.i0)))
                  ? add ?    upwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) == perturbed_predicate<Alpha>(aspoint(arcs,a.i0),aspoint(arcs,a.i1))
                        : rightwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) ^ a.left
                  : perturbed_predicate_two_sqrts<UpwardsA<add>,UpwardsB<0>,UpwardsB<2>,Beta<0,1>,Beta<2,3>>(a.left^add?-1:1,b.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b.i0),aspoint(arcs,b.i1)));
}

// // Are the intersections of two circles with a third counterclockwise?  In other words, is the triangle c0,x01,x02 positively oriented?
// // The two intersections are assumed to exist.
// namespace {
// // ai is true if we're the coefficient of the ith sqrt
// template<bool a0,bool a1> struct Ordered { static Exact<6-2*(a0+a1)> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
//   const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
//   const auto dc1 = c1-c0, dc2 = c2-c0;
//   return choice<a0>(Alpha::eval(S0,S1),One())
//        * choice<a1>(Alpha::eval(S0,S2),One())
//        * small_mul(a0 && !a1 ? -1 : 1, choice<a0!=a1>(edet(dc1,dc2),edot(dc1,dc2)));
// }};
//
//
// }
// static bool circle_intersections_ordered_helper(Arcs arcs, const Vertex v0, const Vertex v1) {
//   assert(v0.i0==v1.i0);
//   // Perform case analysis based on the two quadrants
//   const int q0 = v0.q0,
//             q1 = v1.q0;
//   switch ((q1-q0)&3) {
//     case 0: return circle_intersections_upwards      (arcs,v0,v1) ^ (q0==1 || q0==2);
//     case 2: return circle_intersections_upwards<true>(arcs,v0,v1) ^ (q0==1 || q0==2);
//     case 3: return false;
//     default: return true; // case 1
//   }
// }
// // Tests if an arc segment is less then a half circle
// static bool circle_intersections_ordered(Arcs arcs, const Vertex v0, const Vertex v1) {
//   assert(v0.i0==v1.i0);
//   const Vector<Interval,2> center(arcs[v0.i0].center);
//   return FILTER(cross(v0.p()-center,v1.p()-center),
//                 circle_intersections_ordered_helper(arcs,v0,v1));
// }

// Does the (a1,b) intersection occur on the piece of a1 between a0 and a2?  a1 and b are assumed to intersect.
bool circle_arc_intersects_circle(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a1b) {
  assert(a01.i1==a12.i0 && a12.i0==a1b.i0);
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
bool circle_intersection_inside_circle(Arcs arcs, const Vertex a, const int b) {
  // This predicate has the form
  //   A + B sqrt(Beta<0,1>) < 0
  // where
  struct A { template<class TV> static PredicateType<4,TV> eval(const TV S0, const TV S1, const TV S2) {
    const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
    const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z;
    const auto c01 = c1-c0, c02 = c2-c0;
    const auto sqr_c01 = esqr_magnitude(c01),
               sqr_c02 = esqr_magnitude(c02),
               alpha01 = sqr_c01+(r0+r1)*(r0-r1),
               alpha02 = sqr_c02+(r0+r2)*(r0-r2);
    return alpha01*edot(c01,c02)-alpha02*sqr_c01;
  }};
  struct B { template<class TV> static PredicateType<2,TV> eval(const TV S0, const TV S1, const TV S2) {
    const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
    return edet(c1-c0,c2-c0);
  }};
  return FILTER(sqr(Interval(arcs[b].radius))-sqr_magnitude(a.p()-Vector<Interval,2>(arcs[b].center)),
                perturbed_predicate_sqrt<A,B,Beta<0,1>>(a.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b)));
}

// Is the (a0,a1) intersection to the right of b's center?  Degree 6, but can be eliminated entirely.
bool circle_intersection_right_of_center(Arcs arcs, const Vertex a, const int b) {
  // This predicate has the form
  //   (\hat{alpha} c01.x - 2 c02.x c01^2) - c01.y sqrt(Beta<0,1>)
  struct A { template<class TV> static PredicateType<3,TV> eval(const TV S0, const TV S1, const TV S2) {
    const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
    const auto r0 = S0.z,    r1 = S1.z;
    const auto c01 = c1-c0;
    const auto sqr_c01 = esqr_magnitude(c01),
               alpha = sqr_c01+(r0+r1)*(r0-r1);
    return alpha*c01.x-((c2.x-c0.x)<<1)*sqr_c01;
  }};
  struct B { template<class TV> static PredicateType<1,TV> eval(const TV S0, const TV S1, const TV S2) {
    return S1.y-S0.y;
  }};
  return FILTER(a.p().x-arcs[b].center.x,
                perturbed_predicate_sqrt<A,B,Beta<0,1>>(a.left?-1:1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b)));
}

Array<Vertex> compute_vertices(Arcs arcs, RawArray<const int> next) {
  IntervalScope scope;
  Array<Vertex> vertices(arcs.size(),false); // vertices[i] is the start of arcs[i]
  for (int i0=0;i0<arcs.size();i0++) {
    const int i1 = next[i0];
    vertices[i1] = circle_circle_intersections(arcs,i0,i1)[arcs[i0].left];
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
static bool circle_intersection_below_horizontal(Arcs arcs, const Vertex a01, const HorizontalVertex a0y) {
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
  if (ay.arc==by.arc)
    return ay.left;
  return FILTER(by.x-ay.x,
                perturbed_predicate_two_sqrts<RightwardsA,One,One,RightwardsC<0>,RightwardsC<1>>(ay.left?1:-1,by.left?-1:1,aspoint(arcs,ay.arc),aspoint(arcs,by.arc),aspoint_horizontal(ay.y)));
}

} // namespace other
