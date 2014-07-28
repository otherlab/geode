// Exact geometric predicates

#include <geode/exact/predicates.h>
#include <geode/exact/Exact.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/exact/scope.h>
#include <geode/array/RawArray.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
namespace geode {

using exact::Perturbed;
typedef exact::Perturbed2 P2;
typedef exact::Perturbed3 P3;
using exact::ImplicitlyPerturbed;

// First, a trivial predicate, handled specially so that it can be partially inlined.

template<int axis, class Perturbed> bool axis_less_degenerate(const Perturbed a, const Perturbed b) {
  struct F { static void eval(RawArray<mp_limb_t> result, RawArray<const Vector<Exact<1>,Perturbed::ValueType::m>> X) {
    mpz_set(result,X[1][axis]-X[0][axis]);
  }};
  const Perturbed X[2] = {a,b};
  return perturbed_sign(F::eval,1,asarray(X));
}

#define IAL(d,axis) \
  template bool axis_less_degenerate<axis,Perturbed<d>>(const Perturbed<d>,const Perturbed<d>); \
  template bool axis_less_degenerate<axis,ImplicitlyPerturbed<d>>(const ImplicitlyPerturbed<d>,const ImplicitlyPerturbed<d>);
IAL(2,0) IAL(2,1)
IAL(3,0) IAL(3,1) IAL(3,2)
#undef IAL

template bool axis_less_degenerate<0,exact::ImplicitlyPerturbedCenter>(const exact::ImplicitlyPerturbedCenter,const exact::ImplicitlyPerturbedCenter);
template bool axis_less_degenerate<1,exact::ImplicitlyPerturbedCenter>(const exact::ImplicitlyPerturbedCenter,const exact::ImplicitlyPerturbedCenter);

// Polynomial predicates


namespace {
struct TriangleOriented { template<class TV> static inline PredicateType<2,TV> eval(const TV p0, const TV p1, const TV p2) {
  return edet(p1-p0,p2-p0);
}};}
bool triangle_oriented(const P2 p0, const P2 p1, const P2 p2) {
  return perturbed_predicate<TriangleOriented>(p0,p1,p2);
}

namespace {
struct DirectionsOriented { template<class TV> static inline PredicateType<2,TV> eval(const TV d0, const TV d1) {
  return edet(d0,d1);
}};}
bool directions_oriented(const P2 d0, const P2 d1) {
  return perturbed_predicate<DirectionsOriented>(d0,d1);
}

namespace {
struct SegmentDirectionsOriented { template<class TV> static inline PredicateType<2,TV> eval(const TV a0, const TV a1, const TV b0, const TV b1) {
  return edet(a1-a0,b1-b0);
}};}
bool segment_directions_oriented(const P2 a0, const P2 a1, const P2 b0, const P2 b1) {
  return perturbed_predicate<SegmentDirectionsOriented>(a0,a1,b0,b1);
}

namespace {
struct SegmentToDirectionOriented { template<class TV> static inline PredicateType<2,TV> eval(const TV a0, const TV a1, const TV d) {
  return edet(a1-a0,d);
}};}
bool segment_to_direction_oriented(const P2 a0, const P2 a1, const P2 d) {
  return perturbed_predicate<SegmentToDirectionOriented>(a0,a1,d);
}

namespace {
struct SegmentIntersectionsOrdered { template<class TV> static inline PredicateType<4,TV> eval(const TV a0, const TV a1, const TV b0, const TV b1, const TV c0, const TV c1) {
  const auto da = a1-a0,
             db = b1-b0,
             dc = c1-c0;
  return edet(c0-a0,dc)*edet(da,db)-edet(b0-a0,db)*edet(da,dc);
}};}
bool segment_intersections_ordered(const P2 a0, const P2 a1, const P2 b0, const P2 b1, const P2 c0, const P2 c1) {
  return perturbed_predicate<SegmentIntersectionsOrdered>(a0,a1,b0,b1,c0,c1)
       ^ segment_directions_oriented(a0,a1,b0,b1)
       ^ segment_directions_oriented(a0,a1,c0,c1);
}

namespace {
struct SegmentIntersectionUpwards { template<class TV> static inline PredicateType<3,TV> eval(const TV a0, const TV a1, const TV b0, const TV b1, const TV c0) {
  const auto da = a1-a0,
             db = b1-b0;
             // c1 == c1 + x*delta
             //dc = [dx, 0];
  //return delta*(a0.y-c0.y)*edet(da,db)-edet(b0-a0,db)*da.y*delta;
  return (a0.y-c0.y)*edet(da,db)-edet(b0-a0,db)*da.y;
}};}
bool segment_intersection_upwards(const P2 a0, const P2 a1, const P2 b0, const P2 b1, const P2 c) {
  return perturbed_predicate<SegmentIntersectionUpwards>(a0,a1,b0,b1,c)
       ^ segment_directions_oriented(a0,a1,b0,b1)
       ^ rightwards(a0,a1);
}

#define ROW(d) tuple(esqr_magnitude(d),d.x,d.y) // Put the quadratic entry first so that subexpressions are lower order

namespace {
struct Incircle { template<class TV> static inline PredicateType<4,TV> eval(const TV p0, const TV p1, const TV p2, const TV p3) {
  const auto d0 = p0-p3,
             d1 = p1-p3,
             d2 = p2-p3;
  return edet(ROW(d0),ROW(d1),ROW(d2));
}};}
bool incircle(const P2 p0, const P2 p1, const P2 p2, const P2 p3) {
  return perturbed_predicate<Incircle>(p0,p1,p2,p3);
}

bool segments_intersect(const P2 a0, const P2 a1, const P2 b0, const P2 b1) {
  return triangle_oriented(a0,a1,b0)!=triangle_oriented(a0,a1,b1)
      && triangle_oriented(b0,b1,a0)!=triangle_oriented(b0,b1,a1);
}

namespace {
struct TetrahedronOriented {
  template<class TV> static inline PredicateType<3,TV> eval(const TV p0, const TV p1, const TV p2, const TV p3) {
    return edet(p1-p0,p2-p0,p3-p0);
  }
};}
bool tetrahedron_oriented(const P3 p0, const P3 p1, const P3 p2, const P3 p3) {
  return perturbed_predicate<TetrahedronOriented>(p0,p1,p2,p3);
}

namespace {
struct SegmentTriangleOriented {
  template<class TV> static inline PredicateType<3,TV> eval(const TV a0, const TV a1,
                                                            const TV b0, const TV b1, const TV b2) {
    return edet(b1-b0,b2-b0,a1-a0);
  }
};}
bool segment_triangle_oriented(const P3 a0, const P3 a1, const P3 b0, const P3 b1, const P3 b2) {
  return perturbed_predicate<SegmentTriangleOriented>(a0,a1,b0,b1,b2);
}

bool segment_triangle_intersect(const P3 a0, const P3 a1, const P3 b0, const P3 b1, const P3 b2) {
  if (   tetrahedron_oriented(a0,b0,b1,b2)
      == tetrahedron_oriented(a1,b0,b1,b2))
    return false;
  bool   c01 =  tetrahedron_oriented(a0,a1,b0,b1);
  return c01 == tetrahedron_oriented(a0,a1,b1,b2)
      && c01 == tetrahedron_oriented(a0,a1,b2,b0);
}

namespace {
struct SegmentTriangleIntersectionsOrdered {
  template<class TV> static inline PredicateType<6,TV> eval(const TV a0, const TV a1,
                                                            const TV b0, const TV b1, const TV b2,
                                                            const TV c0, const TV c1, const TV c2) {
    const auto da = a1-a0;
    const auto nb = ecross(b1-b0,b2-b0),
               nc = ecross(c1-c0,c2-c0);
    return edot(da,nb)*edot(c0-a0,nc)-edot(da,nc)*edot(b0-a0,nb);
  }
};}
bool segment_triangle_intersections_ordered(const P3 a0, const P3 a1,
                                            const P3 b0, const P3 b1, const P3 b2,
                                            const P3 c0, const P3 c1, const P3 c2) {
  return perturbed_predicate<SegmentTriangleIntersectionsOrdered>(a0,a1,b0,b1,b2,c0,c1,c2)
       ^ segment_triangle_oriented(a0,a1,b0,b1,b2)
       ^ segment_triangle_oriented(a0,a1,c0,c1,c2);
}

namespace {
struct TrianglesOriented {
  template<class TV> static inline PredicateType<6,TV> eval(const TV a0, const TV a1, const TV a2,
                                                            const TV b0, const TV b1, const TV b2,
                                                            const TV c0, const TV c1, const TV c2) {
    return edet(ecross(a1-a0,a2-a0),
                ecross(b1-b0,b2-b0),
                ecross(c1-c0,c2-c0));
  }
};}
bool triangles_oriented(const P3 a0, const P3 a1, const P3 a2,
                        const P3 b0, const P3 b1, const P3 b2,
                        const P3 c0, const P3 c1, const P3 c2) {
  return perturbed_predicate<TrianglesOriented>(a0,a1,a2,b0,b1,b2,c0,c1,c2);
}

// Unit tests.  Warning: These do not check the geometric correctness of the predicates, only properties of exact computation and perturbation.

static void predicate_tests() {
  IntervalScope scope;
  typedef Vector<double,2> TV2;
  typedef Vector<Quantized,2> QV2;

  // Compare triangle_oriented and incircle against approximate floating point versions
  struct F {
    static inline double triangle_oriented(const TV2 p0, const TV2 p1, const TV2 p2) {
      return edet(p1-p0,p2-p0);
    };
    static inline double incircle(const TV2 p0, const TV2 p1, const TV2 p2, const TV2 p3) {
      const auto d0 = p0-p3, d1 = p1-p3, d2 = p2-p3;
      return edet(ROW(d0),ROW(d1),ROW(d2));
    }
  };
  const auto random = new_<Random>(9817241);
  for (int step=0;step<100;step++) {
    #define MAKE(i) \
      const auto p##i = P2(i,QV2(random->uniform<Vector<ExactInt,2>>(-exact::bound,exact::bound))); \
      const TV2 x##i(p##i.value());
    MAKE(0) MAKE(1) MAKE(2) MAKE(3)
    GEODE_ASSERT(triangle_oriented(p0,p1,p2)==(F::triangle_oriented(x0,x1,x2)>0));
    GEODE_ASSERT(incircle(p0,p1,p2,p3)==(F::incircle(x0,x1,x2,x3)>0));
  }

  // Test behavior for large numbers, using the scale invariance and antisymmetry of incircle.
  for (const int i : range(exact::log_bound)) {
    const auto bound = ExactInt(1)<<i;
    const auto p0 = P2(0,QV2(-bound,-bound)), // Four points on a circle of radius sqrt(2)*bound
               p1 = P2(1,QV2( bound,-bound)),
               p2 = P2(2,QV2( bound, bound)),
               p3 = P2(3,QV2(-bound, bound));
    GEODE_ASSERT(!incircle(p0,p1,p2,p3));
    GEODE_ASSERT( incircle(p0,p1,p3,p2));
  }
}

}
using namespace geode;

void wrap_predicates() {
  GEODE_FUNCTION(predicate_tests)
}
