// Nearly exact geometric constructions

#include <geode/exact/constructions.h>
#include <geode/exact/predicates.h>
#include <geode/exact/perturb.h>
#include <geode/exact/Exact.h>
#include <geode/exact/Interval.h>
#include <geode/exact/math.h>
#include <geode/exact/scope.h>
#include <geode/array/RawArray.h>
#include <geode/geometry/Segment.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cout;
using std::endl;
using exact::Point2;
typedef Vector<Quantized,2> EV2;

namespace {
struct SegmentSegment { template<class TV> static ConstructType<2,3,TV> eval(const TV a0, const TV a1,
                                                                             const TV b0, const TV b1) {
  const auto da = a1-a0,
             db = b1-b0;
  const auto den = edet(da,db);
  const auto num = emul(den,a0)+emul(edet(b0-a0,db),da);
  return tuple(num,den);
}};}
exact::Vec2 segment_segment_intersection(const Point2 a0, const Point2 a1, const Point2 b0, const Point2 b1) {
  // TODO: Use the sign returned by perturbed_construct?
  return perturbed_construct<SegmentSegment>(segment_segment_intersection_threshold,a0,a1,b0,b1).x;
}

static bool check_intersection(const EV2 a0, const EV2 a1, const EV2 b0, const EV2 b1, Random& random) {
  typedef Vector<double,2> DV;
  const int i = random.bits<uint32_t>();
  const Point2 a0p(i,a0), a1p(i+1,a1), b0p(i+2,b0), b1p(i+3,b1);
  if (segments_intersect(a0p,a1p,b0p,b1p)) {
    const auto ab = segment_segment_intersection(a0p,a1p,b0p,b1p);
    GEODE_ASSERT(Segment<DV>(DV(a0),DV(a1)).distance(DV(ab)) < 1.01);
    GEODE_ASSERT(Segment<DV>(DV(b0),DV(b1)).distance(DV(ab)) < 1.01);
    return true;
  }
  return false;
}

static void construction_tests() {
  Log::Scope log("construction tests");
  IntervalScope scope;

  {
    const Point2 a0(0,EV2(-100.1,   0)),
                 a1(1,EV2( 100.1,   0)),
                 b0(2,EV2(   0,-100.1)),
                 b1(3,EV2(   0, 100.1)),

                 c0(4,EV2( 100.2, 200.1)),
                 c1(5,EV2( 300.2, 200.1)),
                 d0(6,EV2( 200.2, 100.1)),
                 d1(7,EV2( 200.2, 300.1));

    const Point2 e0(8,EV2(-100,  50)),
                 e1(9,EV2( 100,  50));

    // Check very simple cases where two segments intersect
    GEODE_ASSERT(segments_intersect(a0,a1,b1,b0));
    GEODE_ASSERT(segments_intersect(c0,c1,d0,d1));
    GEODE_ASSERT(segments_intersect(a0,a1,b0,b1));

    // Check very simple case where two segments don't intersect
    GEODE_ASSERT(!segments_intersect(a0,b0,a1,b1));
    GEODE_ASSERT(!segments_intersect(a0,a1,e0,e1));
  }

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
    GEODE_ASSERT(sqr(count-prob*total)<sqr(3)*total*prob*(1-prob)); // Require that we're within 3 standard deviations.
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
    GEODE_ASSERT(sqr_magnitude(ab-vec(x1,y))<=2);
  }

  // Check exactly colinear segments
  {
    const int total = 450;
    int count = 0;
    for (int k=0;k<total;k++) {
      // Pick a random line with simple rational slope, then choose four random points exactly on this line.
      const EV2 base (random->uniform<Vector<ExactInt,2>>(-1000,1000)),
                slope(random->uniform<Vector<ExactInt,2>>(-15,16));
      const auto ts = random->uniform<Vector<int,4>>(-1000,1000);
      count += check_intersection(base+ts.x*slope,
                                  base+ts.y*slope,
                                  base+ts.z*slope,
                                  base+ts.w*slope,random);
    }
    cout << "colinear: total = "<<total<<", count = "<<count<<endl;
    GEODE_ASSERT(count==100);
  }

  // Finally, check all coincident points
  {
    const int total = 48;
    int count = 0;
    for (int k=0;k<total;k++) {
      const auto p = EV2(perturbation<2>(16,k));
      const Point2 a0(k,p), a1(k+1,p), b0(k+2,p), b1(k+3,p);
      if (segments_intersect(a0,a1,b0,b1)) {
        GEODE_ASSERT(segment_segment_intersection(a0,a1,b0,b1)==p);
        count++;
      }
    }
    cout << "coincident: count = "<<count<<endl;
    GEODE_ASSERT(count==13);
  }
}

}
using namespace geode;

void wrap_constructions() {
  GEODE_FUNCTION(construction_tests)
}
