// Robust constructive solid geometry for circular arc polygons in the plane

#include <other/core/exact/circle_csg.h>
#include <other/core/exact/Interval.h>
#include <other/core/exact/math.h>
#include <other/core/exact/predicates.h>
#include <other/core/exact/perturb.h>
#include <other/core/exact/quantize.h>
#include <other/core/exact/scope.h>
#include <other/core/array/amap.h>
#include <other/core/array/convert.h>
#include <other/core/array/sort.h>
#include <other/core/geometry/BoxTree.h>
#include <other/core/geometry/traverse.h>
#include <other/core/math/One.h>
#include <other/core/math/mean.h>
#include <other/core/python/wrap.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/process.h>
#include <other/core/utility/str.h>
namespace other {

using Log::cout;
using std::endl;
using exact::Exact;
typedef exact::Vec2 EV2;
typedef Vector<Exact<1>,3> LV3;

// If true, always run both fast and slow tests and compare results
// IMPORTANT: This is a much stronger test than the pure unit tests, and should be run whenver this file is changed.
#define CHECK 0

// Unfortunately, circular arc CSG requires a rather large number of predicates, all doing extremely similar things.  In particular,
// most of these routines answer questions about the relative positions of intersections between arcs.  Therefore, we precompute
// these intersections wherever possible, so that most questions can be answered with fast interval queries.  This mechanism also
// allows testing to be integrated into each routine, via a compile time flag that answers a question the fast way *and* the slow way
// and asserts consistency.

namespace {
// A precomputed intersection of two arcs
struct Vertex {
  int i0,i1; // Flat indices of the previous and next circle
  bool left; // True if the intersection is to the left of the (i,j) segment
  uint8_t q0,q1; // Quadrants relative to i0's center and i1's center, respectively
  EV2 inexact; // The nearly exactly rounded intersect, differing from the exact intersection by at most two.

  bool operator==(const Vertex v) const {
    return i0==v.i0 && i1==v.i1 && left==v.left;
  }

  bool operator!=(const Vertex v) const {
    return !(*this==v);
  }

  friend Hash hash_reduce(const Vertex v) {
    return Hash(v.i0,2*v.i1+v.left);
  }

  OTHER_UNUSED friend ostream& operator<<(ostream& output, const Vertex v) {
    return output<<format("(%d,%d,%c)",v.i0,v.i1,v.left?'L':'R');
  }

  // A conservative interval containing the true intersection
  Vector<Interval,2> p() const {
    return Vector<Interval,2>(Interval(inexact.x-2,inexact.x+2),
                              Interval(inexact.y-2,inexact.y+2));
  }

  // The same as p(), but with a different type
  Box<EV2> box() const {
    return Box<EV2>(inexact).thickened(2);
  }

  // Reverse the vertex to go from i1 to i0
  Vertex reverse() const {
    Vertex r;
    r.i0 = i1;
    r.i1 = i0;
    r.left = !left;
    r.q0 = q1;
    r.q1 = q0;
    r.inexact = inexact;
    return r;
  }
};
}

typedef RawArray<const ExactCircleArc> Arcs;
typedef RawArray<const Vector<int,2>> Near; // (prev,next) pairs
typedef RawArray<const Vertex> Vertices;

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

// Cautionary tale: An earlier version of this routine had an option to negate the radius,
// which was used to reduce code duplication in circles_intersect.  This would have been
// an absolute disaster, as wouldn't have flipped the sign of the symbolic perturbation.
static inline exact::Point3 aspoint(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,exact::Vec3(a.center,a.radius));
}
static inline exact::Point2 aspoint_center(Arcs arcs, const int arc) {
  const auto& a = arcs[arc];
  return tuple(a.index,a.center);
}

// Do two circles intersect (degree 2)?
namespace {
template<bool add> struct Intersect { static inline Exact<2> eval(const LV3 S0, const LV3 S1) {
  const auto c0 = S0.xy(), c1 = S1.xy();
  const auto r0 = S0.z,    r1 = S1.z;
  return sqr(add?r1+r0:r1-r0)-esqr_magnitude(c1-c0);
}};}
static bool circles_intersect(Arcs arcs, const int arc0, const int arc1) {
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
static Vector<Vertex,2> circle_circle_intersections(Arcs arcs, const int arc0, const int arc1) {
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
          v.x.inexact = sl-sq;
          v.y.inexact = sl+sq;
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
                 alpha_hat = sqr_dc-(r1+r0)*(r1-r0);
    struct FR { static Vector<Exact<>,3> eval(RawArray<const exact::Vec3> X) {
      MOST
      const auto v = emul(two_sqr_dc,c0)+emul(alpha_hat,dc);
      return Vector<Exact<>,3>(Exact<>(v.x),Exact<>(v.y),Exact<>(two_sqr_dc));
    }};
    struct FS { static Vector<Exact<>,3> eval(RawArray<const exact::Vec3> X) {
      MOST
      const auto sqr_beta_hat = small_mul(4,sqr(r0))*sqr_dc-sqr(alpha_hat);
      return Vector<Exact<>,3>(sqr_beta_hat*sqr(dc.x),sqr_beta_hat*sqr(dc.y),sqr(two_sqr_dc));
    }};
    #undef MOST
    const exact::Point3 X[2] = {aspoint(arcs,arc0),aspoint(arcs,arc1)};
    const auto fr = perturbed_ratio(&FR::eval,3,asarray(X)),
               fs = rotate_left_90(perturbed_ratio(&FS::eval,6,asarray(X),true)*EV2(axis_less<0>(X[0],X[1])?1:-1,
                                                                                    axis_less<1>(X[0],X[1])?1:-1));
#if CHECK
    OTHER_ASSERT(   check_linear.x.thickened(1).contains(fr.x)
                 && check_linear.y.thickened(1).contains(fr.y));
    OTHER_ASSERT(   check_quadratic.x.thickened(1).contains(fs.x)
                 && check_quadratic.y.thickened(1).contains(fs.y));
#endif
    v.x.inexact = fr - fs;
    v.y.inexact = fr + fs;
  }

  // Fill in quadrants
quadrants:
  v.x.q0 = circle_circle_intersection_quadrant(arcs,v.x);
  v.x.q1 = circle_circle_intersection_quadrant(arcs,v.x.reverse());
  v.y.q0 = circle_circle_intersection_quadrant(arcs,v.y);
  v.y.q1 = circle_circle_intersection_quadrant(arcs,v.y.reverse());
  return v;
}

// Is intersection (a0,a1).y < (b0,b1).y?  If add = true, assume a0=b0 and check whether ((0,a1)+(0,b1)).y > 0.
// This is degree 20 as written, but can be reduced to 6.
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
template<bool add=false> static bool circle_intersections_upwards(Arcs arcs, const Vertex a, const Vertex b) {
  assert(a!=b && a!=b.reverse() && (!add || a.i0==b.i0));
  return FILTER(add ? a.p().y+b.p().y-(arcs[a.i0].center.y<<1) : b.p().y-a.p().y,
                (a.i0==b.i0 && a.i1==b.i1) || (a.i0==b.i1 && a.i1==b.i0) // If we're comparing the two intersections of the same pair of circles, use special code to avoid zero polynomials in perturbation
                  ? add ?    upwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) == perturbed_predicate<Alpha>(aspoint(arcs,a.i0),aspoint(arcs,a.i1))
                        : rightwards(aspoint_center(arcs,a.i0),aspoint_center(arcs,a.i1)) ^ a.left
                  : perturbed_predicate_two_sqrts<UpwardsA<add>,UpwardsB<0>,UpwardsB<2>,Beta<0,1>,Beta<2,3>>(a.left^add?-1:1,b.left?1:-1,aspoint(arcs,a.i0),aspoint(arcs,a.i1),aspoint(arcs,b.i0),aspoint(arcs,b.i1)));
}

// Are the intersections of two circles with a third counterclockwise?  In other words, is the triangle c0,x01,x02 positively oriented?
// The two intersections are assumed to exist.
namespace {
// ai is true if we're the coefficient of the ith sqrt
template<bool a0,bool a1> struct Ordered { static Exact<6-2*(a0+a1)> eval(const LV3 S0, const LV3 S1, const LV3 S2) {
  const auto c0 = S0.xy(), c1 = S1.xy(), c2 = S2.xy();
  const auto dc1 = c1-c0, dc2 = c2-c0;
  return choice<a0>(Alpha::eval(S0,S1),One())
       * choice<a1>(Alpha::eval(S0,S2),One())
       * small_mul(a0 && !a1 ? -1 : 1, choice<a0!=a1>(edet(dc1,dc2),edot(dc1,dc2)));
}};}
static bool circle_intersections_ordered_helper(Arcs arcs, const Vertex v0, const Vertex v1) {
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
static bool circle_intersections_ordered(Arcs arcs, const Vertex v0, const Vertex v1) {
  assert(v0.i0==v1.i0);
  const Vector<Interval,2> center(arcs[v0.i0].center);
  return FILTER(cross(v0.p()-center,v1.p()-center),
                circle_intersections_ordered_helper(arcs,v0,v1));
}

// Does the (a1,b) intersection occur on the piece of a1 between a0 and a2?  a1 and b are assumed to intersect.
static bool circle_arc_intersects_circle(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a1b) {
  const bool small = circle_intersections_ordered(arcs,a01.reverse(),a12);
  return !arcs[a01.i1].positive ^ small ^ (   (small^ circle_intersections_ordered(arcs,a01.reverse(),a1b))
                                           || (small^!circle_intersections_ordered(arcs,a12          ,a1b)));
}

// Does the piece of a1 between a0 and a1 intersect the piece of b1 between b0 and b2?  a1 and b1 are assumed to intersect.
static bool circle_arcs_intersect(Arcs arcs, const Vertex a01, const Vertex a12,
                                             const Vertex b01, const Vertex b12,
                                             const Vertex ab) {
  return circle_arc_intersects_circle(arcs,a01,a12,ab)
      && circle_arc_intersects_circle(arcs,b01,b12,ab.reverse());
}

// Is the (a0,a1) intersection inside circle b?  Degree 8, but can be eliminated entirely.
static bool circle_intersection_inside_circle(Arcs arcs, const Vertex a, const int b) {
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
static bool circle_intersection_right_of_center(Arcs arcs, const Vertex a, const int b) {
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

// Compute winding(local_outside) - winding(rightwards), where local_outside is immediately outside of a12 and rightwards
// is far to the right of a12, taking into account only arcs a1 and a2.  Thus, ignoring secondary intersections with arcs a1 and a2,
// the result will be either 0 or -1, since locally winding(local_outside) = 0 and winding(rightwards) = 0 or 1.
static int local_x_axis_depth(Arcs arcs, const Vertex a01, const Vertex a12, const Vertex a23) {
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
static int horizontal_depth_change(Arcs arcs, const Vertex a, const Vertex b01, const Vertex b12) {
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

static Array<Box<EV2>> arc_boxes(Near near, Arcs arcs, RawArray<const Vertex> vertices) {
  // Build bounding boxes for each arc
  Array<Box<EV2>> boxes(arcs.size(),false);
  for (int i1=0;i1<arcs.size();i1++) {
    const int i2 = near[i1].y;
    const auto v01 = vertices[i1],
               v12 = vertices[i2];
    auto box = bounding_box(v01.inexact,v12.inexact).thickened(2);
    int q0 = v01.q1,
        q1 = v12.q0;
    if (q0==q1) {
      if (arcs[i1].positive != circle_intersections_ordered(arcs,v01.reverse(),v12)) {
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
    boxes[i1] = box;
  }
  return boxes;
}

Nested<ExactCircleArc> exact_split_circle_arcs(Nested<const ExactCircleArc> nested, const int depth) {
  // Check input consistency
  for (const int p : range(nested.size())) {
    const auto contour = nested[p];
    if (contour.size()==2 && contour[0].left!=contour[1].left)
      throw RuntimeError(format("exact_split_circle_arcs: contour %d is degenerate of size 2",p));
    for (const auto& arc : contour)
      OTHER_ASSERT(arc.radius>0,"Radii must be positive so that symbolic perturbation doesn't make them negative");
  }

  // Prepare for interval arithmetic
  IntervalScope scope;
  Arcs arcs = nested.flat;

  // Build a convenience array of (prev,next) pairs to avoid dealing with the nested structure.
  // Also precompute all intersections between connected arcs.
  Array<Vector<int,2>> near(arcs.size(),false);
  for (int i=1;i<near.size();i++) {
    near[i].x = i-1;
    near[i-1].y = i;
  }
  for (int p=0;p<nested.size();p++) {
    const int lo = nested.offsets[p],
              hi = nested.offsets[p+1];
    if (lo < hi) {
      near[lo].x = hi-1;
      near[hi-1].y = lo;
    }
  }

  // Precompute all intersections between connected arcs
  Array<Vertex> vertices(arcs.size(),false); // vertices[i] is the start of arcs[i]
  for (int i1=0;i1<arcs.size();i1++) {
    const int i0 = near[i1].x;
    vertices[i1] = circle_circle_intersections(arcs,i0,i1)[arcs[i0].left];
  }

  // Compute all nontrivial intersections between segments
  struct Intersections {
    const BoxTree<EV2>& tree;
    Near near;
    Arcs arcs;
    Vertices vertices;
    Array<Vertex> found;

    Intersections(const BoxTree<EV2>& tree, Near near, Arcs arcs, Vertices vertices)
      : tree(tree), near(near), arcs(arcs), vertices(vertices) {}

    bool cull(const int n) const { return false; }
    bool cull(const int n0, const int box1) const { return false; }
    void leaf(const int n) const { assert(tree.prims(n).size()==1); }

    void leaf(const int n0, const int n1) {
      if (n0 != n1) {
        assert(tree.prims(n0).size()==1 && tree.prims(n1).size()==1);
        const int i1 = tree.prims(n0)[0], i2 = near[i1].y,
                  j1 = tree.prims(n1)[0], j2 = near[j1].y;
        if (   !(i2==j1 && j2==i1) // If we're looking at the two arcs of a length two contour, there's nothing to do
            && (i1==j2 || i2==j1 || circles_intersect(arcs,i1,j1))) {
          // We can get here even if the two arcs are adjacent, since we may need to detect the other intersection of two adjacent circles.
          const auto a01 = vertices[i1],
                     a12 = vertices[i2],
                     b01 = vertices[j1],
                     b12 = vertices[j2];
          for (const auto ab : circle_circle_intersections(arcs,i1,j1))
            if (   ab!=a01.reverse() && ab!=a12 && ab!=b01 && ab!=b12.reverse()
                && circle_arcs_intersect(arcs,a01,a12,b01,b12,ab))
              found.append(ab);
        }
      }
    }
  };
  const auto tree = new_<BoxTree<EV2>>(arc_boxes(near,arcs,vertices),1);
  Intersections pairs(tree,near,arcs,vertices);
  double_traverse(*tree,pairs);

  // Group intersections by segment.  Each pair is added twice: once for each order.
  Array<int> counts(arcs.size());
  for (auto v : pairs.found) {
    counts[v.i0]++;
    counts[v.i1]++;
  }
  Nested<Vertex> others(counts,false); // Invariant: if v in others[i], v.i0 = i.  This implies some wasted space, unfortunately.
  for (auto v : pairs.found) {
    others(v.i0,--counts[v.i0]) = v;
    others(v.i1,--counts[v.i1]) = v.reverse();
  }
  pairs.found.clean_memory();
  counts.clean_memory();

  // Walk all original polygons, recording which subsegments occur in the final result
  Hashtable<Vertex,Vertex> graph; // If u -> v, the output contains the portion of segment j from ij_a to jk_b
  for (const int p : range(nested.size())) {
    const auto poly = range(nested.offsets[p],nested.offsets[p+1]);
    // Compute the depth of the first point in the contour by firing a ray along the positive x axis.
    struct Depth {
      const BoxTree<EV2>& tree;
      Near near;
      Arcs arcs;
      Vertices vertices;
      const Vertex start;
      int depth;

      Depth(const BoxTree<EV2>& tree, Near near, Arcs arcs, Vertices vertices, const int i0, const int i1)
        : tree(tree), near(near), arcs(arcs), vertices(vertices)
        , start(vertices[i1])
        // If we intersect no other arcs, the depth depends on the orientation of direction = (1,0) relative to inwards and outwards arcs
        , depth(local_x_axis_depth(arcs,vertices[i0],start,vertices[near[i1].y])) {}

      bool cull(const int n) const {
        const auto box = tree.boxes(n),
                   sbox = start.box();
        return box.max.x<sbox.min.x || box.max.y<sbox.min.y || box.min.y>sbox.max.y;
      }

      void leaf(const int n) {
        assert(tree.prims(n).size()==1);
        const int j = tree.prims(n)[0];
        if (start.i0!=j && start.i1!=j)
          depth -= horizontal_depth_change(arcs,start,vertices[j],
                                                      vertices[near[j].y]);
      }
    };
    Depth ray(tree,near,arcs,vertices,poly.back(),poly[0]);
    single_traverse(*tree,ray);

    // Walk around the contour, recording all subarcs at the desired depth
    int delta = ray.depth-depth;
    Vertex prev = vertices[poly[0]];
    for (const int i : poly) {
      const auto other = others[i];
      // Sort intersections along this segment
      if (other.size() > 1) {
        struct PairOrder {
          Near near;
          Arcs arcs;
          const Vertex start; // The start of the segment

          PairOrder(Near near, Arcs arcs, Vertex start)
            : near(near), arcs(arcs)
            , start(start) {}

          bool operator()(const Vertex b0, const Vertex b1) const {
            assert(start.i1==b0.i0 && b0.i0==b1.i0);
            if (b0.i1==b1.i1 && b0.left==b1.left)
              return false;
            return circle_arc_intersects_circle(arcs,start,b1,b0);
          }
        };
        sort(other,PairOrder(near,arcs,prev));
      }
      // Walk through each intersection of this segment, updating delta as we go and remembering the subsegment if it has the right depth
      for (const auto o : other) {
        if (!delta)
          graph.set(prev,o);
        delta += o.left ^ arcs[o.i0].positive ^ arcs[o.i1].positive ? -1 : 1;
        prev = o.reverse();
      }
      // Advance to the next segment
      const auto next = vertices[near[i].y];
      if (!delta)
        graph.set(prev,next);
      prev = next;
    }
  }

  // Walk the graph to produce output polygons
  Hashtable<Vertex> seen;
  Nested<ExactCircleArc,false> output;
  for (const auto& it : graph) {
    const auto start = it.key;
    if (seen.set(start)) {
      auto v = start;
      for (;;) {
        auto a = arcs[v.i0];
        a.left = v.left;
        output.flat.append(a);
        v = graph.get(v);
        if (v == start)
          break;
        seen.set(v);
      }
      output.offsets.append(output.flat.size());
    }
  }
  return output;
}

Tuple<Quantizer<real,2>,Nested<ExactCircleArc>> quantize_circle_arcs(Nested<const CircleArc> input) {
  // Compute an approximate bounding box for all arcs
  Box<Vector<real,2>> box;
  for (const auto poly : input)
    for (int j=0,i=poly.size()-1;j<poly.size();i=j++)
      box.enlarge(bounding_box(poly[i].x,poly[j].x).thickened(.5*abs(poly[i].q)*magnitude(poly[i].x-poly[j].x)));

  // Enlarge box quite a lot so that we can closely approximate lines.
  // The error in approximating a straight segment of length L by a circular arc of radius R is
  //   er = L^2/(8*R)
  // If the maximum radius is R, the error due to quantization is
  //   eq = R/bound
  // Equating these, we get
  //   R/bound = L^2/(8*R)
  //   R^2 = L^2*bound/8
  //   R = L*sqrt(bound/8)
  const real max_radius = sqrt(exact::bound/8)*box.sizes().max();
  const auto quant = quantizer(box.thickened(max_radius));

  // Quantize and implicitize each arc
  IntervalScope scope;
  auto output = Nested<ExactCircleArc>::empty_like(input);
  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    const auto out = output[p];
    const int n = in.size();
    for (int j=0,i=n-1;j<n;i=j++) {
      // Implicitize
      const auto x0 = in[i].x,
                 x1 = in[j].x,
                 dx = x1-x0;
      const auto L = magnitude(dx);
      const auto q = in[i].q;
      // Compute radius, quantize, then compute center from quantized radius to reduce endpoint error
      ExactCircleArc e;
      e.radius = max(ExactInt(1),ExactInt(round(quant.scale*min(.25*L*abs(q+1/q),max_radius))),ExactInt(ceil(.5*quant.scale*L)));
      const auto radius = quant.inverse.inv_scale*e.radius;
      const auto center = L ? .5*(x0+x1)+((q>0)^(abs(q)>1)?1:-1)*sqrt(max(0.,sqr(radius/L)-.25))*rotate_left_90(dx) : x0;
      e.center = quant(center);
      e.index = base+i;
      e.positive = q > 0;
      out[i] = e;
    }
    // Fill in left flags
    for (int j=0,i=n-1;j<n;i=j++) {
      const auto x = in[j].x,
                 c0 = quant.inverse(out[i].center),
                 c1 = quant.inverse(out[j].center);
      out[i].left = cross(c1-c0,x-c0)>0;
    }
  }

  // Tweak quantized circles so that they intersect.
  // Also fill in left flags to get as close as possible to the original endpoint.
  for (const int p : range(input.size())) {
    const auto in = input[p];
    const auto out = output[p];
    const int n = in.size();
    // TODO: For now, we require nonequal centers
    for (int j=0,i=n-1;j<n;i=j++)
      OTHER_ASSERT(out[i].center != out[j].center);
    const auto save = out.copy();
    // Iteratively enlarge radii until we're nondegenerately intersecting.  TODO: Currently, this is worst case O(n^2).
    int iters = 0;
    for (;;) {
      bool done = true;
      for (int j=0,i=n-1;j<n;i=j++) {
        const double dc = magnitude(Vector<double,2>(out[i].center-out[j].center));
        ExactInt &ri = out[i].radius,
                 &rj = out[j].radius;
        if (ri+rj <= dc) {
          const auto d = ExactInt(floor((dc-ri-rj)/2+1));
          ri += d;
          rj += d;
          done = false;
        }
        if (abs(ri-rj) >= dc) {
          (ri<rj?ri:rj) = max(ri,rj)-ExactInt(ceil(dc-1));
          done = false;
        }
      }
      if (done)
        break;
      iters++;
    }
  }
  return tuple(quant,output);
}

Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2> quant, Nested<const ExactCircleArc> input) {
  IntervalScope scope;
  const auto output = Nested<CircleArc>::empty_like(input);
  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    const auto out = output[p];
    const int n = in.size();
    for (int j=0,i=n-1;j<n;i=j++)
      out[j].x = quant.inverse(circle_circle_intersections(input.flat,base+i,base+j)[in[i].left].inexact);
    for (int j=0,i=n-1;j<n;i=j++) {
      const auto x0 = out[i].x,
                 x1 = out[j].x,
                 c = quant.inverse(in[i].center);
      const auto radius = quant.inverse.inv_scale*in[i].radius;
      const auto half_L = .5*magnitude(x1-x0);
      const int s = in[i].positive ^ (cross(x1-x0,c-x0)>0) ? -1 : 1;
      out[i].q = half_L/(radius+s*sqrt(max(0.,sqr(radius)-sqr(half_L)))) * (in[i].positive ? 1 : -1);
    }
  }
  return output;
}

Nested<CircleArc> split_circle_arcs(Nested<const CircleArc> arcs, const int depth) {
  const auto e = quantize_circle_arcs(arcs);
  return unquantize_circle_arcs(e.x,exact_split_circle_arcs(e.y,depth));
}

ostream& operator<<(ostream& output, const CircleArc& arc) {
  return output << format("CircleArc([%g,%g],%g)",arc.x.x,arc.x.y,arc.q);
}

ostream& operator<<(ostream& output, const ExactCircleArc& arc) {
  return output << format("ExactCircleArc([%d,%d],%d,%c%c)",arc.center.x,arc.center.y,arc.radius,arc.positive?'+':'-',arc.left?'L':'R');
}

#ifdef OTHER_PYTHON

// Instantiate Python conversions for arrays of circular arcs
namespace {
template<> struct NumpyDescr<CircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){OTHER_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<CircleArc>:public mpl::true_{};
template<> struct NumpyRank<CircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<CircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<CircleArc>::d;
template<> struct NumpyDescr<ExactCircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){OTHER_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<ExactCircleArc>:public mpl::true_{};
template<> struct NumpyRank<ExactCircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<ExactCircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<ExactCircleArc>::d;
}
ARRAY_CONVERSIONS(1,CircleArc)
ARRAY_CONVERSIONS(1,ExactCircleArc)

static void _set_circle_arc_dtypes(PyObject* inexact, PyObject* exact) {
  OTHER_ASSERT(PyArray_DescrCheck(inexact));
  OTHER_ASSERT(PyArray_DescrCheck(exact));
  OTHER_ASSERT(((PyArray_Descr*)inexact)->elsize==sizeof(CircleArc));
  OTHER_ASSERT(((PyArray_Descr*)  exact)->elsize==sizeof(ExactCircleArc));
  Py_INCREF(inexact);
  Py_INCREF(  exact);
  NumpyDescr<     CircleArc>::d = (PyArray_Descr*)inexact;
  NumpyDescr<ExactCircleArc>::d = (PyArray_Descr*)  exact;
}

static Nested<CircleArc> circle_arc_quantize_test(Nested<const CircleArc> arcs) {
  const auto e = quantize_circle_arcs(arcs);
  return unquantize_circle_arcs(e.x,e.y);
}

// The area between a segment of length 2 and an associated circular sector
static inline double q_factor(double q) {
  // Economized rational approximation courtesy of Mathematica.  I suppose this is a tiny binary blob?
  const double qq = q*q;
  return abs(q)<.25 ? q*(1.3804964920832707+qq*(1.018989299316004+0.14953934953934955*qq))/(1.035372369061972+qq*(0.5571675010595465+1./33*qq))
                    : .5*(atan(q)*sqr((1+qq)/q)-(1-qq)/q);
}

real circle_arc_area(RawArray<const CircleArc> arcs) {
  const int n = arcs.size();
  real area = 0;
  for (int i=n-1,j=0;j<n;i=j++)
    area += .5*cross(arcs[i].x,arcs[j].x) + .25*sqr_magnitude(arcs[j].x-arcs[i].x)*q_factor(arcs[i].q); // Triangle area plus circular sector area
  return .5*area;
}

real circle_arc_area(Nested<const CircleArc> polys) {
  real area = 0;
  for (const auto arcs : polys)
    area += circle_arc_area(arcs);
  return area;
}

Nested<CircleArc> canonicalize_circle_arcs(Nested<const CircleArc> polys) {
  // Find the minimal point in each polygon under lexicographic order
  Array<int> mins(polys.size());
  for (int p=0;p<polys.size();p++) {
    const auto poly = polys[p];
    for (int i=1;i<poly.size();i++)
      if (lex_less(poly[i].x,poly[mins[p]].x))
        mins[p] = i;
  }

  // Sort the polygons
  struct Order {
    Nested<const CircleArc> polys;
    RawArray<const int> mins;
    Order(Nested<const CircleArc> polys, RawArray<const int> mins)
      : polys(polys), mins(mins) {}
    bool operator()(int i,int j) const {
      return lex_less(polys(i,mins[i]).x,polys(j,mins[j]).x);
    }
  };
  Array<int> order = arange(polys.size()).copy();
  sort(order,Order(polys,mins));

  // Copy into new array
  Nested<CircleArc> new_polys(polys.sizes().subset(order).copy(),false);
  for (int p=0;p<polys.size();p++) {
    const int base = mins[order[p]];
    const auto poly = polys[order[p]];
    const auto new_poly = new_polys[p];
    for (int i=0;i<poly.size();i++)
      new_poly[i] = poly[(i+base)%poly.size()];
  }
  return new_polys;
}

#endif

}
using namespace other;

void wrap_circle_csg() {
  OTHER_FUNCTION(split_circle_arcs)
  OTHER_FUNCTION(canonicalize_circle_arcs)
  OTHER_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
#ifdef OTHER_PYTHON
  OTHER_FUNCTION(_set_circle_arc_dtypes)
  OTHER_FUNCTION(circle_arc_quantize_test)
#endif
}
