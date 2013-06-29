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
#define CHECK 1

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


bool arcs_from_same_circle(const RawArray<const ExactCircleArc>& arcs, int i0, int i1) {
  if(i0 == i1) return true;
  const auto& a0 = arcs[i0], a1 = arcs[i1];
  if(a0.index != a1.index) return false;
  assert(a0.center == a1.center && a0.radius == a1.radius && a0.positive == a1.positive);
  return true;
}

// Do two circles intersect (degree 2)?
namespace {
template<bool add> struct Intersect { static inline Exact<2> eval(const LV3 S0, const LV3 S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return sqr(add?r1+r0:r1-r0)-esqr_magnitude(c1-c0);
}};}
bool circles_intersect(Arcs arcs, const int arc0, const int arc1) {
  return     perturbed_predicate<Intersect<true >>(aspoint(arcs,arc0),aspoint(arcs,arc1))
         && !perturbed_predicate<Intersect<false>>(aspoint(arcs,arc0),aspoint(arcs,arc1));
}

namespace {
struct Alpha { static Exact<2> eval(const LV3 S0, const LV3 S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return esqr_magnitude(c1-c0)+(r0+r1)*(r0-r1);
}};
template<int i,int j> struct Beta { template<class... Args> static Exact<4> eval(const Args... args) {
  const auto S = tuple(args...);
  const auto& Si = S.template get<i>();
  const auto& Sj = S.template get<j>();
  const auto c0 = Si.xy(), c1 = Sj.xy();
  const auto r0 = Si.z,    r1 = Sj.z;
  const auto sqr_dc = esqr_magnitude(c1-c0);
  return small_mul(4,sqr(r0))*sqr_dc-sqr(sqr_dc+(r0+r1)*(r0-r1));
}};
}

// Which quadrant is in the intersection of two circles in relative to the center of the first?
// The quadrants are 0 to 3 counterclockwise from positive/positive.
// This function should be used only from circle_circle_intersections, where it is precomputed as Vertex::q0.
// As written this is degree 4, but it can be reduced to degree 2 if necessary.
namespace {
template<int axis> struct QuadrantA { static Exact<3> eval(const LV3 S0, const LV3 S1) {
  return Alpha::eval(S0,S1)*(axis==0?S1.y-S0.y:S0.x-S1.x);
}};
template<int axis> struct QuadrantB { static Exact<1> eval(const LV3 S0, const LV3 S1) {
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
                 two_sqr_dc = small_mul(2,sqr_dc), \
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
      const auto sqr_beta_hat = small_mul(4,sqr(r0))*sqr_dc-sqr(alpha_hat);
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

// Is intersection (a0,a1).y < (b0,b1).y?  This is degree 20 as written, but can be reduced to 6.
// If add = true, assume a0=b0 and check whether ((0,a1)+(0,b1)).y > 0.
namespace {
template<bool add> struct UpwardsA { static Exact<5> eval(const LV3 S0, const LV3 S1, const LV3 S2, const LV3 S3) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy(), c3 = S3.xy();
  const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z,    r3 = S3.z;
  const auto c01 = c1-c0, c23 = c3-c2;
  const auto sqr_c01 = esqr_magnitude(c01),
             sqr_c23 = esqr_magnitude(c23);
  const auto alpha01 = sqr_c01+(r0+r1)*(r0-r1),
             alpha23 = sqr_c23+(r2+r3)*(r2-r3);
  return !add ? small_mul(2,c2.y-c0.y)*sqr_c01*sqr_c23+alpha23*(c23.y*sqr_c01)-alpha01*(c01.y*sqr_c23)
              : alpha23*(c23.y*sqr_c01)+alpha01*(c01.y*sqr_c23);
}};
template<int i> struct UpwardsB { static Exact<3> eval(const LV3 S0, const LV3 S1, const LV3 S2, const LV3 S3) {
  BOOST_STATIC_ASSERT(i==0 || i==2);
  const auto c01 = S1.xy()-S0.xy(),
             c23 = S3.xy()-S2.xy();
  return i==0 ? c01.x*esqr_magnitude(c23) // Negated below
              : c23.x*esqr_magnitude(c01);
}};}
template<bool add=false> bool circle_intersections_upwards_helper(Arcs arcs, const Vertex a, const Vertex b) {
  assert(a!=b && a!=b.reverse() && (!add || a.i0==b.i0));
  return FILTER(add ? a.p().y+b.p().y-(arcs[a.i0].center.y*2) : b.p().y-a.p().y,
               (    (arcs_from_same_circle(arcs, a.i0, b.i0) && arcs_from_same_circle(arcs, a.i1, b.i1))
                 || (arcs_from_same_circle(arcs, a.i0, b.i1) && arcs_from_same_circle(arcs, a.i1, b.i0)))
                  ? add ?    upwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) == perturbed_predicate<Alpha>(aspoint(arcs,a.i0),aspoint(arcs,a.i1))
                        : rightwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) ^ a.left
                  : perturbed_predicate_two_sqrts<UpwardsA<add>,UpwardsB<0>,UpwardsB<2>,Beta<0,1>,Beta<2,3>>(a.left^add?-1:1,b.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b.i0),aspoint(arcs,b.i1)));
}
bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b) {
  return circle_intersections_upwards_helper(arcs, a, b);
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
  struct A { static Exact<4> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
    const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
    const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z;
    const auto c01 = c1-c0, c02 = c2-c0;
    const auto sqr_c01 = esqr_magnitude(c01),
               sqr_c02 = esqr_magnitude(c02),
               alpha01 = sqr_c01+(r0+r1)*(r0-r1),
               alpha02 = sqr_c02+(r0+r2)*(r0-r2);
    return alpha01*edot(c01,c02)-alpha02*sqr_c01;
  }};
  struct B { static Exact<2> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
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
  struct A { static Exact<3> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
    const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
    const auto r0 = S0.z,    r1 = S1.z;
    const auto c01 = c1-c0;
    const auto sqr_c01 = esqr_magnitude(c01),
               alpha = sqr_c01+(r0+r1)*(r0-r1);
    return alpha*c01.x-small_mul(2,c2.x-c0.x)*sqr_c01;
  }};
  struct B { static Exact<1> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
    return S1.y-S0.y;
  }};
  return FILTER(a.p().x-arcs[b].center.x,
                perturbed_predicate_sqrt<A,B,Beta<0,1>>(a.left?-1:1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b)));
}

Array<Vertex> compute_verticies(RawArray<const ExactCircleArc> arcs, RawArray<const int> next) {
  IntervalScope scope;
  Array<Vertex> vertices(arcs.size(),false); // vertices[i] is the start of arcs[i]
  for (int i0=0;i0<arcs.size();i0++) {
    const int i1 = next[i0];
    vertices[i1] = circle_circle_intersections(arcs,i0,i1)[arcs[i0].left];
  }
  return vertices;
}

// Compute winding(local_outside) - winding(rightwards), where local_outside is immediately outside of a12 and rightwards
// is far to the right of a12, taking into account only arcs a1 and a2.  Thus, ignoring secondary intersections with arcs a1 and a2,
// the result will be either 0 or -1, since locally winding(local_outside) = 0 and winding(rightwards) = 0 or 1.
int local_x_axis_depth(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a23) {
  assert(a01.i1==a12.i0 && a12.i1==a23.i0);
  // Compute quadrants of both centers and the differentials going in and out of the intersection.
  const bool a1_positive = arcs[a12.i0].positive,
             a2_positive = arcs[a23.i0].positive;
  const int q1 = a12.q0,
            q2 = a12.q1,
            q_in  = (q1+3+2*a1_positive)&3, // The quadrant of the a1 arc differential pointing towards x12
            q_out = (q2+3+2*a2_positive)&3;
  // Compute the depth contribution due to the neighborhood of (a1,a2).  If we come in below horizontal and leave above the
  // result is 0, and it is -1 for the above to below case since then rightwards is slightly inside the arc.  Otherwise, we
  // come in and head out on the same side of the horizontal line, and the result depends on the orientation of the inwards
  // and outwards tangents: it is -1 if we make a right turn at a12 (since then rightwards is inside), 0 if we make a left
  // turn (since then rightwards is outside).  Since we make a right turn iff !a12.left, the result is
  const int local = -(q_in/2==q_out/2 ? q_out/2!=0
                                      : !a12.left ^ a1_positive ^ a2_positive);

  // Unlike in the straight line polygon case, arcs a1,a2 can make additional depth contributions through their other intersections.
  // The existence of such contributions depends on (1) whether the arc goes up or down, (2) whether the center is to the left or right of x01
  // and (3) whether the other intersection is above or below the horizontal line.
  const int near01 = ((q1==1 || q1==2) && (q_in>=2)!=circle_intersections_upwards(arcs,a12,a01)) * (q_in <2 ? -1 : 1),
            near23 = ((q2==1 || q2==2) && (q_out<2)!=circle_intersections_upwards(arcs,a12,a23)) * (q_out<2 ? -1 : 1);
  return local+near01+near23;
}

// Count the depth change along the horizontal ray from (a0,a1) to (a0,a1+(inf,0) due to the arc from (b0,b1) to (b1,b2).
// The change is -1 if we go out of an arc, +1 if we go into an arc.  Degree 8 as written, but can be eliminated entirely.
namespace {
template<int rsign> struct HorizontalA { static Exact<3> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
  const auto r0 = S0.z,    r1 = S1.z,    r2 = S2.z;
  const auto c01 = c1-c0;
  const auto sqr_c01 = esqr_magnitude(c01);
  return (sqr_c01+(r0+r1)*(r0-r1))*c01.y-small_mul(2,c2.y-c0.y+(rsign>0?r2:-r2))*sqr_c01;
}};}
int horizontal_depth_change(Arcs arcs, const Vertex a, const Vertex b01, const Vertex b12) {
  assert(b01.i1==b12.i0);
  const int b1 = b12.i0;
  // Does the horizontal line intersect circle b1?  If not, the depth change is zero.
  struct B { static Exact<1> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
    return S1.x-S0.x;
  }};
  if (   FILTER(a.p().y-(arcs[b1].center.y+arcs[b1].radius),
                 perturbed_predicate_sqrt<HorizontalA<+1>,B,Beta<0,1>>(a.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b1)))
      || FILTER((arcs[b1].center.y-arcs[b1].radius)-a.p().y,
                !perturbed_predicate_sqrt<HorizontalA<-1>,B,Beta<0,1>>(a.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b1))))
    return 0;

  // Determine whether b01 and b12 are above the horizontal line
  const bool b01_above = circle_intersections_upwards(arcs,a,b01),
             b12_above = circle_intersections_upwards(arcs,a,b12);
  const bool b1_positive = arcs[b1].positive;
  if (b01_above != b12_above) {
    // The (b0,b1,b2) arc intersects the horizontal line exactly once.  We first determine whether this intersection is to the right of bc1 = b1.center.
    const bool bh_right_of_bc1 = b01_above ^ b1_positive;
    // Next, we compute whether x01 lies to the right of c2
    const bool x01_right_of_bc1 = circle_intersection_right_of_center(arcs,a,b1);
    // If these differ we are done, otherwise the result depends on whether x01 is inside b1
    return (b12_above?-1:1) * (bh_right_of_bc1 != x01_right_of_bc1 ? bh_right_of_bc1
                                                                   : bh_right_of_bc1 ^ !circle_intersection_inside_circle(arcs,a,b1));
  } else {
    // The (b0,b1,b2) arc intersects the horizontal line either zero or two times.  First, we rule out zero times.
    const int q0 = b01.q1,
              q2 = b12.q0,
              shift = b01_above ? 1 : 3, // Shift so that the vertical axis oriented towards the horizontal line becomes the positive x-axis
              sq0 = (q0+shift)&3,
              sq2 = (q2+shift)&3;
    const bool zero_intersections = !b1_positive ^ (q0 != q2 ? sq2 > sq0
                                                             : circle_intersections_upwards(arcs,b01.reverse(),b12) ^ (q0==1 || q0==2));
    if (zero_intersections)
      return 0;
    // If both intersections occur to the same side of x01.x, there's no depth change
    if (!circle_intersection_inside_circle(arcs,a,b1))
      return 0;
    // A depth change!
    return b1_positive ? -1 : 1;
  }
}


} // namespace other