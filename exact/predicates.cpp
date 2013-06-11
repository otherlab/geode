// Exact geometric predicates

#include <other/core/exact/predicates.h>
#include <other/core/exact/Exact.h>
#include <other/core/exact/math.h>
#include <other/core/exact/perturb.h>
#include <other/core/array/RawArray.h>
#include <other/core/python/wrap.h>
#include <other/core/random/Random.h>
namespace other {

using std::cout;
using std::endl;
using exact::Point;
using exact::Point2;
using exact::Exact;
typedef Vector<Exact<1>,2> LV2;
typedef Vector<Exact<1>,3> LV3;

// First, a trivial predicate, handled specially so that it can be partially inlined.

template<int axis,int d> bool axis_less_degenerate(const Tuple<int,Vector<Quantized,d>> a, const Tuple<int,Vector<Quantized,d>> b) {
  struct F { static inline Exact<> eval(RawArray<const Vector<Exact<1>,d>> X) {
    return Exact<>(X[1][axis]-X[0][axis]);
  }};
  const typename Point<d>::type X[2] = {a,b};
  return perturbed_sign(F::eval,1,asarray(X));
}

#define IAL(d,axis) template bool axis_less_degenerate<axis,d>(const Tuple<int,Vector<Quantized,d>>,const Tuple<int,Vector<Quantized,d>>);
IAL(2,0) IAL(2,1)
IAL(3,0) IAL(3,1) IAL(3,2)

// Polynomial predicates

bool triangle_oriented(const Point2 p0, const Point2 p1, const Point2 p2) {
  struct F { static inline Exact<2> eval(const LV2 p0, const LV2 p1, const LV2 p2) {
    return edet(p1-p0,p2-p0);
  }};
  return perturbed_predicate<F>(p0,p1,p2);
}

bool directions_oriented(const Point2 d0, const Point2 d1) {
  struct F { static inline Exact<2> eval(const LV2 d0, const LV2 d1) {
    return edet(d0,d1);
  }};
  return perturbed_predicate<F>(d0,d1);
}

bool segment_directions_oriented(const Point2 a0, const Point2 a1, const Point2 b0, const Point2 b1) {
  struct F { static inline Exact<2> eval(const LV2 a0, const LV2 a1, const LV2 b0, const LV2 b1) {
    return edet(a1-a0,b1-b0);
  }};
  return perturbed_predicate<F>(a0,a1,b0,b1);
}

bool segment_to_direction_oriented(const Point2 a0, const Point2 a1, const Point2 d) {
  struct F { static inline Exact<2> eval(const LV2 a0, const LV2 a1, const LV2 d) {
    return edet(a1-a0,d);
  }};
  return perturbed_predicate<F>(a0,a1,d);
}

bool segment_intersections_ordered(const Point2 a0, const Point2 a1, Point2 b0, Point2 b1, Point2 c0, Point2 c1) {
  struct F { static inline Exact<4> eval(const LV2 a0, const LV2 a1, const LV2 b0, const LV2 b1, const LV2 c0, const LV2 c1) {
    const auto da = a1-a0,
               db = b1-b0,
               dc = c1-c0;
    return edet(c0-a0,dc)*edet(da,db)-edet(b0-a0,db)*edet(da,dc);
  }};
  return perturbed_predicate<F>(a0,a1,b0,b1,c0,c1)
       ^ segment_directions_oriented(a0,a1,b0,b1)
       ^ segment_directions_oriented(a0,a1,c0,c1);
}

bool incircle(const Point2 p0, const Point2 p1, const Point2 p2, const Point2 p3) {
  struct F { static inline Exact<4> eval(const LV2 p0, const LV2 p1, const LV2 p2, const LV2 p3) {
    const auto d0 = p0-p3,
               d1 = p1-p3,
               d2 = p2-p3;
    #define ROW(d) tuple(esqr_magnitude(d),d.x,d.y) // Put the quadratic entry first so that subexpressions are lower order
    return edet(ROW(d0),ROW(d1),ROW(d2));
  }};
  return perturbed_predicate<F>(p0,p1,p2,p3);
}

bool segments_intersect(const exact::Point2 a0, const exact::Point2 a1, const exact::Point2 b0, const exact::Point2 b1) {
  return triangle_oriented(a0,a1,b0)!=triangle_oriented(a0,a1,b1)
      && triangle_oriented(b0,b1,a0)!=triangle_oriented(b0,b1,a1);
}

// Unit tests.  Warning: These do not check the geometric correctness of the predicates, only properties of exact computation and perturbation.

static void predicate_tests() {
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
      const auto p##i = tuple(i,QV2(random->uniform<Vector<ExactInt,2>>(-exact::bound,exact::bound))); \
      const TV2 x##i(p##i.y);
    MAKE(0) MAKE(1) MAKE(2) MAKE(3)
    OTHER_ASSERT(triangle_oriented(p0,p1,p2)==(F::triangle_oriented(x0,x1,x2)>0));
    OTHER_ASSERT(incircle(p0,p1,p2,p3)==(F::incircle(x0,x1,x2,x3)>0));
  }

  // Test behavior for large numbers, using the scale invariance and antisymmetry of incircle.
  for (const int i : range(exact::log_bound)) {
    const auto bound = ExactInt(1)<<i;
    const auto p0 = tuple(0,QV2(-bound,-bound)), // Four points on a circle of radius sqrt(2)*bound
               p1 = tuple(1,QV2( bound,-bound)),
               p2 = tuple(2,QV2( bound, bound)),
               p3 = tuple(3,QV2(-bound, bound));
    OTHER_ASSERT(!incircle(p0,p1,p2,p3));
    OTHER_ASSERT( incircle(p0,p1,p3,p2));
  }
}

}
using namespace other;

void wrap_predicates() {
  OTHER_FUNCTION(predicate_tests)
}
