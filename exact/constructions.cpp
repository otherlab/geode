// Nearly exact geometric constructions

#include <other/core/exact/constructions.h>
#include <other/core/exact/predicates.h>
#include <other/core/exact/perturb.h>
#include <other/core/exact/Exact.h>
#include <other/core/exact/Interval.h>
#include <other/core/exact/math.h>
#include <other/core/exact/scope.h>
#include <other/core/array/RawArray.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/python/wrap.h>
#include <other/core/random/Random.h>
#include <other/core/utility/Log.h>
namespace other {

using Log::cout;
using std::endl;
using exact::Exact;
using exact::Point2;
typedef Vector<Quantized,2> EV2;

exact::Vec2 segment_segment_intersection(const Point2 a0, const Point2 a1, const Point2 b0, const Point2 b1) {
  // Evaluate conservatively using intervals
  {
    const Vector<Interval,2> a0i(a0.y), a1i(a1.y), b0i(b0.y), b1i(b1.y);
    const auto da = a1i-a0i,
               db = b1i-b0i;
    const auto den = edet(da,db);
    if (!den.contains_zero()) {
      const auto r = a0i+edet(b0i-a0i,db)*da*inverse(den);
      if (small(r,segment_segment_intersection_threshold))
        return snap(r);
    }
  }

  // If intervals fail, evaluate and round exactly using symbolic perturbation
  struct F { static Vector<Exact<>,3> eval(RawArray<const Vector<Exact<1>,2>> X) {
    const auto a0(X[0]), a1(X[1]), b0(X[2]), b1(X[3]);
    const auto da = a1-a0,
               db = b1-b0;
    const auto den = edet(da,db);
    return Vector<Exact<>,3>(Vector<Exact<>,2>(emul(den,a0)+emul(edet(b0-a0,db),da)),Exact<>(den));
  }};
  const Point2 X[4] = {a0,a1,b0,b1};
  return perturbed_ratio(&F::eval,3,asarray(X));
}

static bool check_intersection(const EV2 a0, const EV2 a1, const EV2 b0, const EV2 b1, Random& random) {
  typedef Vector<double,2> DV;
  const int i = random.bits<uint32_t>();
  const Point2 a0p(i,a0), a1p(i+1,a1), b0p(i+2,b0), b1p(i+3,b1);
  if (segments_intersect(a0p,a1p,b0p,b1p)) {
    const auto ab = segment_segment_intersection(a0p,a1p,b0p,b1p);
    OTHER_ASSERT(Segment<DV>(DV(a0),DV(a1)).distance(DV(ab)) < 1.01);
    OTHER_ASSERT(Segment<DV>(DV(b0),DV(b1)).distance(DV(ab)) < 1.01);
    return true;
  }
  return false;
}

static void construction_tests() {
  Log::Scope log("construction tests");
  IntervalScope scope;

  // Check a bunch of large random segments
  const auto random = new_<Random>(623189131);
  {
    const int total = 4096;
    int count = 0;
    for (int k=0;k<total;k++)
      count += check_intersection(EV2(perturbation<2>(4,k)),
                                  EV2(perturbation<2>(5,k)),
                                  EV2(perturbation<2>(6,k)),
                                  EV2(perturbation<2>(7,k)),random);
    // See https://groups.google.com/forum/?fromgroups=#!topic/sci.math.research/kRvImz5RslU
    const double prob = 25./108;
    cout << "random: expected "<<round(prob*total)<<"+-"<<round(sqrt(total*prob*(1-prob)))<<", got "<<count<<endl;
    OTHER_ASSERT(sqr(count-prob*total)<sqr(3)*total*prob*(1-prob)); // Require that we're within 3 standard deviations.
  }

  // Check nearly colinear segments
  for (int k=0;k<32;k++) {
    const auto med = exact::bound/16,
               big = exact::bound/2;
    const Quantized x1 = random->uniform<ExactInt>(-med,med),
                    x0 = random->uniform<ExactInt>(-big,x1),
                    x2 = random->uniform<ExactInt>(x1,big),
                    dx = random->uniform<ExactInt>(-big,big),
                    y  = random->uniform<ExactInt>(-med,med);
    EV2 a0(x0,y),
        a1(x2,y),
        b0(x1-dx,y-1),
        b1(x1+dx,y+1);
    if (random->bit()) swap(a0,a1);
    if (random->bit()) swap(b0,b1);
    if (random->bit()) { swap(a0,b0); swap(a1,b1); }
    const auto ab = segment_segment_intersection(tuple(0,a0),tuple(1,a1),tuple(2,b0),tuple(3,b1));
    // We know the intersection exactly by construction; check that we get it almost exactly right.
    OTHER_ASSERT(sqr_magnitude(ab-vec(x1,y))<=2);
  }

  // Check exactly colinear segments
  {
    const int total = 450;
    int count = 0;
    for (int k=0;k<total;k++) {
      // Pick a random line with simple rational slope, then choose four random points exactly on this line.
      const EV2 base(random->uniform<Vector<ExactInt,2>>(-1000,1000)),
                slope(random->uniform<Vector<ExactInt,2>>(-15,16));
      const auto ts = random->uniform<Vector<int,4>>(-1000,1000);
      count += check_intersection(base+ts.x*slope,
                                  base+ts.y*slope,
                                  base+ts.z*slope,
                                  base+ts.w*slope,random);
    }
    cout << "colinear: total = "<<total<<", count = "<<count<<endl;
    OTHER_ASSERT(count==100);
  }

  // Finally, check all coincident points
  {
    const int total = 48;
    int count = 0;
    for (int k=0;k<total;k++) {
      const auto p = EV2(perturbation<2>(16,k));
      const Point2 a0(k,p), a1(k+1,p), b0(k+2,p), b1(k+3,p);
      if (segments_intersect(a0,a1,b0,b1)) {
        OTHER_ASSERT(segment_segment_intersection(a0,a1,b0,b1)==p);
        count++;
      }
    }
    cout << "coincident: count = "<<count<<endl;
    OTHER_ASSERT(count==13);
  }
}

}
using namespace other;

void wrap_constructions() {
  OTHER_FUNCTION(construction_tests)
}
