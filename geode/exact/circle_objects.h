#pragma once
#include <geode/array/SmallArray.h>
#include <geode/exact/config.h>
#include <geode/exact/forward.h>
#include <geode/exact/Interval.h>
#include <geode/geometry/Box.h>

namespace geode {
////////////////////////////////////////////////////////////////////////////////
// Non-member circle predicates

// Identity checks
// Although perturbation normally resolves degenerate cases, it doesn't help if we attempt to compare a construct to itself
// Unless otherwise noted, these must be used to detect duplicated arguments before calling into other predicates
template<Pb PS> bool is_same_circle      (const            ExactCircle<PS>& c0, const            ExactCircle<PS>& c1);
template<Pb PS> bool is_same_horizontal  (const        ExactHorizontal<PS>& h0, const        ExactHorizontal<PS>& h1);
template<Pb PS> bool is_same_intersection(const  CircleIntersectionKey<PS>& i0, const  CircleIntersectionKey<PS>& i1);
template<Pb PS> bool is_same_intersection(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1); // Should only be used for intersections on the same horizontal

// Boolean predicates
template<Pb PS> bool has_intersections(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1); // Requires caller to ensure c0 != c1
template<Pb PS> bool has_intersections(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1);

template<Pb PS> bool intersections_rightwards(const HorizontalIntersection<PS>& i0, const HorizontalIntersection<PS>& i1); // Should only be used for intersections on the same horizontal

// Does interior of two circles intersect (i.e. Do circles intersect or is one fully inside another)?
template<Pb PS> bool circles_overlap(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1); // Requires c0 != c1

// Check if two arcs on the same circle have non-infinitesimal overlap (i.g. not just sharing an endpoint)
// For use when arcs are on the same circle (i.g. is_same_circle(a0.circle, a1.circle) is true)
template<Pb PS> bool arcs_overlap(const ExactArc<PS>& a0, const ExactArc<PS>& a1);

////////////////////////////////////////////////////////////////////////////////
// Intersections

// Versions of intersections_if_any will always safely ignore degenerate configurations so that caller doesn't need to check inputs
// These check only for 'normal' intersections that can be represented by a CircleIntersection or HorizontalIntersection coincident geometry is ignored (i.g. intersections_if_any(c0,c0).empty() == true)
// Due to symbolic perturbation, circle-circle and circle-horizontal intersections will always have zero or two intersections
// Arcs are treated as open intervals causing intersections exactly at their endpoints to be ignored unless it is a full circle
template<Pb PS> SmallArray<CircleIntersection<PS>,2>     intersections_if_any(const ExactCircle<PS>& c0, const     ExactCircle<PS>& c1);
template<Pb PS> SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1);
template<Pb PS> SmallArray<CircleIntersection<PS>,2>     intersections_if_any(const    ExactArc<PS>& a0, const        ExactArc<PS>& a1);
template<Pb PS> SmallArray<HorizontalIntersection<PS>,2> intersections_if_any(const    ExactArc<PS>& a0, const ExactHorizontal<PS>& h1);

// If caller knows that primitives must intersect due to a construction or other reason, can call these:
template<Pb PS> Vector<CircleIntersection<PS>,2>     get_intersections(const ExactCircle<PS>& c0, const ExactCircle<PS>& c1);
template<Pb PS> Vector<HorizontalIntersection<PS>,2> get_intersections(const ExactCircle<PS>& c0, const ExactHorizontal<PS>& h1);

template<Pb PS> inline Box<exact::Vec2> bounding_box(const ExactCircle<PS>& c);
template<Pb PS> Box<exact::Vec2> bounding_box(const ExactArc<PS>& a);
template<Pb PS> Box<exact::Vec2> bounding_box(const ExactHorizontalArc<PS>& a);

// Test if intersection is on the left side of center of reference circle
template<Pb PS> inline bool left_of_center(const IncidentCircle<PS>& i);
template<Pb PS> inline bool left_of_center(const IncidentHorizontal<PS>& i);

// Overloads for sorting and hashing.
template<Pb PS> inline bool operator==(const ExactCircle<PS>& lhs, const ExactCircle<PS>& rhs) { return is_same_circle(lhs, rhs); }
template<Pb PS> inline bool operator==(const CircleIntersectionKey<PS>& lhs, const CircleIntersectionKey<PS>& rhs) { return is_same_intersection(lhs, rhs); }
template<Pb PS> inline bool operator==(const HorizontalIntersection<PS>& lhs, const HorizontalIntersection<PS>& rhs) { return is_same_intersection(lhs, rhs); }
template<Pb PS> inline bool operator<(const HorizontalIntersection<PS>& lhs, const HorizontalIntersection<PS>& rhs);
template<Pb PS> inline Hash hash_reduce(const ExactCircle<PS>& c);
template<Pb PS> inline Hash hash_reduce(const CircleIntersectionKey<PS>& k);

////////////////////////////////////////////////////////////////////////////////
// For CircleIntersection we use the order of "cl" and "cr" to disambiguate the two intersections between them. (i.g. {c0,c1} is one intersection and {c1,c0} is the other)
// We disambiguate incident intersections around a reference circle, by tracking if the reference circle would become cl or cr in a CircleIntersection
enum class ReferenceSide : bool { cl = false, cr = true };
inline ReferenceSide opposite(const ReferenceSide side) { return static_cast<ReferenceSide>(static_cast<bool>(side)^true); }
// ReferenceSide can be confusing; these are here to help document usage
inline bool cl_is_reference(const ReferenceSide side) { return side == ReferenceSide::cl; }
inline bool cr_is_reference(const ReferenceSide side) { return side == ReferenceSide::cr; }
inline bool cl_is_incident(const ReferenceSide side) { return side == ReferenceSide::cr; }
////////////////////////////////////////////////////////////////////////////////

// Approximate circle intersections are initially computed with interval arithmetic only falling back to exact values when tolerance limits aren't met
// Keeping the computed interval results in larger objects, but can provide tighter bounds that avoid expensive perturbation evaluations later
#ifndef GEODE_KEEP_INTERSECTION_INTERVALS
// Keeping the intervals appears to be about 10% faster
#define GEODE_KEEP_INTERSECTION_INTERVALS 1
#endif

struct ApproxIntersection {
#if GEODE_KEEP_INTERSECTION_INTERVALS
  Vector<Interval,2> _approx_interval;
#else
  exact::Vec2 _rounded = exact::Vec2::repeat(std::numeric_limits<Quantized>::quiet_NaN());
#endif

  inline exact::Vec2 guess() const; // Best estimate for actual value.
  inline exact::Vec2 snapped() const; // Rounded value of guess
  static Quantized tolerance() { return 1; } // Maximum difference (per component) of 'guess' and the true intersection

  inline Vector<Interval,2> p() const; // A conservative interval containing the true intersection
  inline Box<exact::Vec2> box() const; // The same as p(), but with a different type
};

////////////////////////////////////////////////////////////////////////////////

// Depending on perturbation we might or might not need to track an index
// We use template specialization and rely on empty base optimization to ensure implicit perturbation's layout isn't encumbered
template<Pb PS> struct ExactCirclePerturbationHelper;
template<> struct ExactCirclePerturbationHelper<Pb::Implicit> { };
template<> struct ExactCirclePerturbationHelper<Pb::Explicit> {
  int index; // Must be >= 0
  ExactCirclePerturbationHelper() = default;
  ExactCirclePerturbationHelper(const int _index) : index(_index) { }
};

////////////////////////////////////////////////////////////////////////////////

template<Pb PS> struct ExactCircle : public ExactCirclePerturbationHelper<PS> {
  Vector<Quantized,2> center;
  Quantized radius;

  ExactCircle() = default;
  ExactCircle(const ExactCircle<PS>&) = default;
  ExactCircle(const Vector<Quantized,2> _center, const Quantized _radius); // Use iif using Implicit perturbation
  ExactCircle(const Vector<Quantized,2> _center, const Quantized _radius, const int _index); // Use iif using Explicit perturbation

  SmallArray<IncidentCircle<PS>,2>     intersections_if_any(const ExactCircle<PS>& incident) const; // Will return empty result if is_same_circle(*this, incident)
  SmallArray<IncidentHorizontal<PS>,2> intersections_if_any(const ExactHorizontal<PS>& h) const;

  Vector<IncidentCircle<PS>,2>     get_intersections(const ExactCircle<PS>& incident) const;
  Vector<IncidentHorizontal<PS>,2> get_intersections(const ExactHorizontal<PS>& h) const;

  // Find the other intersection point for the same two circles
  IncidentCircle<PS> other_intersection(const IncidentCircle<PS>& i) const;

  // The ccw arc around 'this' from intersection_min to intersection_max is the arc inside 'incident'
  // Calling intersections_if_any or get_intersections should be preferred unless a construction needs a specific intersection only
  IncidentCircle<PS> inline intersection_min(const ExactCircle<PS>& incident) const;
  IncidentCircle<PS> inline intersection_max(const ExactCircle<PS>& incident) const;

  // IncidentCircles allow us to comparing multiple intersections that we know are on this circle
  // Caller is responsible for using is_same_intersection to catch ties
  bool is_same_intersection           (const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const;
  bool intersections_upwards          (const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const;
  bool intersections_upwards          (const IncidentCircle<PS>& i,  const IncidentHorizontal<PS>& h) const;
  bool intersections_ccw              (const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const; // Is the triangle {center,i0,i1} positively oriented?

  // These help make IncidentCircle and IncidentHorizontal interchangable in some templates:
  static GEODE_CONSTEXPR_UNLESS_MSVC bool is_same_intersection(const IncidentCircle<PS>& i, const IncidentHorizontal<PS>& h) { return false; }
  static GEODE_CONSTEXPR_UNLESS_MSVC bool is_same_intersection(const IncidentHorizontal<PS>& h, const IncidentCircle<PS>& i) { return false; }

  // If quadrants of incident circles have already been compared, these can be used
  bool intersections_upwards_same_q   (const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const;
  bool intersections_ccw_same_q       (const IncidentCircle<PS>& i0, const IncidentCircle<PS>& i1) const;
  bool intersections_ccw_same_q       (const IncidentCircle<PS>& i0, const IncidentHorizontal<PS>& i1) const;
  bool intersections_ccw_same_q       (const IncidentHorizontal<PS>& i0, const IncidentCircle<PS>& i1) const { return !intersections_ccw_same_q(i1,i0); }

  // Sort order is ccw with positive x axis first
  template<class TA, class TB> inline bool intersections_sorted(const TA& i0, const TB& i1) const { return (i0.q != i1.q) ? (i0.q < i1.q) : !is_same_intersection(i0,i1) && intersections_ccw_same_q(i0, i1); }
  // unique_intersections_sorted is slightly faster, but caller is responsible for catching ties
  template<class TA, class TB> inline bool unique_intersections_sorted(const TA& i0, const TB& i1) const { assert(!is_same_intersection(i0,i1)); return (i0.q != i1.q) ? (i0.q < i1.q) : intersections_ccw_same_q(i0, i1); }

  // Get angle (in range 0 to 2*pi) of vector from circle center to i relative to positive x axis
  // Warning: These aren't currently used by circle csg and haven't been thoroughly tested. They also rely on evaluation of trig functions that might be comparatively slow
  Interval approx_angle(const IncidentCircle<PS>& i) const;
  Interval approx_angle(const IncidentHorizontal<PS>& i) const;
};
// Check that we didn't break layout
static_assert(sizeof(ExactCircleIm) == 3*sizeof(Quantized), "Error: Implicitly perturbed ExactCircle not using empty base optimization");

////////////////////////////////////////////////////////////////////////////////
// Caches data about an 'incident' circle and its intersection with a 'reference' circle (assumed to be available elsewhere)
template<Pb PS> struct IncidentCircle : public ExactCircle<PS> {
  ApproxIntersection approx;
  uint8_t q; // quadrant of intersection relative to center of reference circle
  ReferenceSide side;

  const ExactCircle<PS>& as_circle() const { return *this; } // Explicit downcast to try and minimize risk of using wrong overloaded function

  // Any values not given will be computed
  IncidentCircle() = default;
  IncidentCircle(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ReferenceSide _side);
  IncidentCircle(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ReferenceSide _side, const ApproxIntersection _approx);
  inline IncidentCircle(const ExactCircle<PS>& incident, const ReferenceSide _side, const ApproxIntersection _approx, const uint8_t _q);
  Box<exact::Vec2> box() const { return approx.box(); }
  Vector<Interval,2> p() const { return approx.p(); }

  // Get the same intersection location using this as the reference
  // i.e. a simpler version of 'CircleIntersection(reference, *this).as_incident_to(this->as_circle())'
  // Warning: Each call to this will recompute intersection quadrant from opposite circle
  IncidentCircle reference_as_incident(const ExactCircle<PS>& reference) const;
};

////////////////////////////////////////////////////////////////////////////////

// Minimum data to unambiguously identify an intersection between two circles
// Does not cache information about the intersection needed by predicates
template<Pb PS> struct CircleIntersectionKey {
  ExactCircle<PS> cl,cr; // The intersecting circles (to the left and right respectively when facing line between their centers)
                         // The intersection is a point to the right of the line from center of cl to center of cr)
  CircleIntersectionKey() = default;
  CircleIntersectionKey(const ExactCircle<PS>& reference, const IncidentCircle<PS>& incident);
 protected:
  // This is protected to ensure use of reference/incident constructor doesn't inadvertently overload to this
  friend struct CircleIntersection<PS>; // Should only be used from CircleIntersection
  CircleIntersectionKey(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr);
 public:
  bool contains(const ExactCircle<PS>& c) const { return is_same_circle(c, cl) || is_same_circle(c, cr); }
  inline ReferenceSide find(const ExactCircle<PS>& reference) const;
  const ExactCircle<PS>& reference(const ReferenceSide side) const { return cl_is_reference(side) ? cl : cr; }
};

// CircleIntersection stores a pair of circles and approximate data about one of their two intersections
// For either ReferenceSide, this can be converted into an ExactCircle and an IncidentCircle without recomputing any predicates
template<Pb PS> struct CircleIntersection : public CircleIntersectionKey<PS> {
  ApproxIntersection approx; // The approximate intersection point
  uint8_t ql,qr; // Quadrants relative to cl's center and cr's center, respectively

  CircleIntersection() = default;
  CircleIntersection(const CircleIntersectionKey<PS>& k, const ApproxIntersection _approx, const uint8_t _ql, const uint8_t _qr);
  // Remaining constructors compute any missing approximate data
  explicit CircleIntersection(const CircleIntersectionKey<PS>& k);
  CircleIntersection(const CircleIntersectionKey<PS>& k, const ApproxIntersection approx);
  CircleIntersection(const ExactCircle<PS>& reference, const IncidentCircle<PS>& incident);
  CircleIntersection(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr, const ApproxIntersection approx);

  static inline CircleIntersection first(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr);

  const CircleIntersectionKey<PS>& as_key() const { return *this; } // Explicit downcast to make usage clearer

  // Get the IncidentCircle to go with CircleIntersectionKey::reference(side)
  inline IncidentCircle<PS> incident(const ReferenceSide side) const;
  IncidentCircle<PS> as_incident_to(const ExactCircle<PS>& reference) const { return incident(this->find(reference)); }

  // Test if an intersection point is inside some unrelated circle
  // Requires c != cl or cr
  bool is_inside(const ExactCircle<PS>& c) const;
};

////////////////////////////////////////////////////////////////////////////////

// Horizontal line along some y value.
template<Pb PS> struct ExactHorizontal {
  Quantized y; // y-coordinate of horizontal line
  ExactHorizontal() = default;
  ExactHorizontal(const Quantized _y) : y(_y) { }
};

template<Pb PS> struct IncidentHorizontal {
  ExactHorizontal<PS> line;  // horizontal line for intersection
  Interval x; // x-coordinate of intersection
  bool left; // True if the intersection is the left of the *vertical* line through the circle's center
  uint8_t q; // Quadrant relative to circle's center
  Vector<Interval,2> p() const { return vec(x,Interval(line.y)); }
  Box<Vec2> box() const { const auto bx = x.box(); return Box<Vec2>(Vec2(bx.min,line.y),Vec2(bx.max,line.y)); }
};

template<Pb PS> struct HorizontalIntersection : public IncidentHorizontal<PS> {
  ExactCircle<PS> circle;
  HorizontalIntersection() = default;
  HorizontalIntersection(const IncidentHorizontal<PS>& i, const ExactCircle<PS>& _circle)
   : IncidentHorizontal<PS>(i)
   , circle(_circle)
  { }
};

////////////////////////////////////////////////////////////////////////////////
// ExactArc represents a non-empty subset of points on a circle (possibly the improper subset 'all')
template<Pb PS> struct ExactArc {
  ExactCircle<PS> circle;
  IncidentCircle<PS> src, dst; // Arc contains all points CCW from src until dst
  // Note: ExactArc doesn't encode an order for its points. Swapping the endpoints forms the complement (unless arc is a full circle).

  bool is_full_circle() const;

  inline bool has_endpoint(const IncidentCircle<PS>& i) const; // Check if i is src or dst
  bool     unsafe_contains(const IncidentCircle<PS>& i) const; // i must not be an endpoint of this arc
  bool   interior_contains(const IncidentCircle<PS>& i) const; // if i is an endpoint result will be false, unless arc is a full circle
  bool  half_open_contains(const IncidentCircle<PS>& i) const; // Check if i is in [src, dst) (will be true for a full circle)
  bool contains_horizontal(const IncidentHorizontal<PS>& h) const; // Since arc starts and ends at incident circles, we don't need to handle coincident endpoints

  // Overloads that automatically convert CircleIntersection to IncidentCircle
  bool      has_endpoint(const CircleIntersection<PS>& i) const { return      has_endpoint(i.as_incident_to(circle)); }
  bool   unsafe_contains(const CircleIntersection<PS>& i) const { return   unsafe_contains(i.as_incident_to(circle)); }
  bool interior_contains(const CircleIntersection<PS>& i) const { return interior_contains(i.as_incident_to(circle)); }

  // Compute q values for a quantized arc accounting for perturbations
  // Returns both q value for the arc and for the opposite arc on the same circle
  // One of the returned values will have magnitude <= 1 and the other will have magnitude >= 1
  // opp_q should be -1/q, but this function will correctly handle sign if q is 0
  // Requires !is_full_circle()
  Vec2 q_and_opp_q() const;
  real q() const { return q_and_opp_q()[0]; }
};

////////////////////////////////////////////////////////////////////////////////

// Like ExactArc, but with one of the endpoints at intersections of a horizontal and a circle
// With some carefully designed templates and overloads this could share most of its implementation with ExactArc, but for now it doesn't seem worth the added complexity
template<Pb PS> struct ExactHorizontalArc {
  ExactCircle<PS> circle;
  IncidentCircle<PS> i;
  IncidentHorizontal<PS> h;
  bool h_is_src;

  // Construct ccw arc from i to h
  ExactHorizontalArc(const ExactCircle<PS>& _circle, const IncidentCircle<PS>& _i, const IncidentHorizontal<PS>& _h)
   : circle(_circle), i(_i), h(_h)
   , h_is_src(false) { }

  // Construct ccw arc from h to i
  ExactHorizontalArc(const ExactCircle<PS>& _circle, const IncidentHorizontal<PS>& _h, const IncidentCircle<PS>& _i)
   : circle(_circle), i(_i), h(_h)
   , h_is_src(true) { }

  bool contains(const IncidentCircle<PS>& other) const;

  bool contains(const CircleIntersection<PS>& other) const { return contains(other.as_incident_to(circle)); }

  SmallArray<IncidentCircle<PS>,2> intersections_if_any(const ExactCircle<PS>& incident) const;
  SmallArray<IncidentCircle<PS>,2> intersections_if_any(const ExactArc<PS>& a) const;
};


////////////////////////////////////////////////////////////////////////////////
// Definitions for inline functions

template<Pb PS> inline bool left_of_center(const IncidentCircle<PS>& i) { return i.q == 1 || i.q == 2; }
template<Pb PS> inline bool left_of_center(const IncidentHorizontal<PS>& i) { return i.q == 1 || i.q == 2; }

template<Pb PS> inline bool operator<(const HorizontalIntersection<PS>& lhs, const HorizontalIntersection<PS>& rhs) {
  return !(lhs == rhs) && intersections_rightwards(lhs, rhs); // We have to check for equality since std::sort will sometimes compare elements with themselves
}
template<> inline Hash hash_reduce(const ExactCircle<Pb::Implicit>& c) { return Hash(c.center,c.radius); }
template<> inline Hash hash_reduce(const ExactCircle<Pb::Explicit>& c) { return Hash(c.index); }
template<Pb PS> inline Hash hash_reduce(const CircleIntersectionKey<PS>& k) { return Hash(k.cl,k.cr); }

template<Pb PS> inline Box<exact::Vec2> bounding_box(const ExactCircle<PS>& c) {
  return Box<Vec2>(c.center).thickened(c.radius); // This is safe since center and radius are integers so result should be exactly representable
}

template<Pb PS> inline bool is_same_intersection(const CircleIntersection<PS>& i0, const CircleIntersection<PS>& i1) {
  const bool result = is_same_intersection(i0.as_key(), i1.as_key());
  assert(!result || (i0.ql == i1.ql && i0.qr == i1.qr));
  assert(!result || i0.approx.box().intersects(i1.approx.box()));
  return result;
}

template<Pb PS> inline CircleIntersection<PS> CircleIntersection<PS>::first(const ExactCircle<PS>& cl, const ExactCircle<PS>& cr) {
  return CircleIntersection<PS>(CircleIntersectionKey<PS>(cl,cr));
}

template<Pb PS> inline IncidentCircle<PS> ExactCircle<PS>::intersection_min(const ExactCircle<PS>& incident) const {
  return IncidentCircle<PS>(*this, incident, ReferenceSide::cl);
}
template<Pb PS> inline IncidentCircle<PS> ExactCircle<PS>::intersection_max(const ExactCircle<PS>& incident) const {
  return IncidentCircle<PS>(incident, *this, ReferenceSide::cr);
}

template<Pb PS> inline IncidentCircle<PS> CircleIntersection<PS>::incident(const ReferenceSide side) const {
  return cl_is_incident(side) ? IncidentCircle<PS>(this->cl, side, approx, qr)
                              : IncidentCircle<PS>(this->cr, side, approx, ql);
}

template<Pb PS> inline IncidentCircle<PS>::IncidentCircle(const ExactCircle<PS>& _incident, const ReferenceSide _side, const ApproxIntersection _approx, const uint8_t _q)
 : ExactCircle<PS>(_incident)
 , approx(_approx)
 , q(_q)
 , side(_side)
{ }

#if GEODE_KEEP_INTERSECTION_INTERVALS
inline exact::Vec2 ApproxIntersection::guess() const { return center(_approx_interval); }
inline exact::Vec2 ApproxIntersection::snapped() const { return snap(_approx_interval); }
inline Vector<Interval,2> ApproxIntersection::p() const { return _approx_interval; }
inline Box<exact::Vec2> ApproxIntersection::box() const { return bounding_box(_approx_interval); }
#else
inline exact::Vec2 ApproxIntersection::guess() const { return _rounded;}
inline exact::Vec2 ApproxIntersection::snapped() const { return _rounded;}
inline Vector<Interval,2> ApproxIntersection::p() const {
  // We can compute differences here since we know that values are exactly representable
  return Vector<Interval,2>(Interval(_rounded.x-tolerance(),_rounded.x+tolerance()),
                            Interval(_rounded.y-tolerance(),_rounded.y+tolerance()));
}
inline Box<exact::Vec2> ApproxIntersection::box() const {
    return Box<exact::Vec2>(_rounded).thickened(tolerance());
}
#endif

template<Pb PS> inline ReferenceSide CircleIntersectionKey<PS>::find(const ExactCircle<PS>& reference) const {
  assert(contains(reference));
  return is_same_circle(reference, cl) ?  ReferenceSide::cl : ReferenceSide::cr;
}

template<Pb PS> inline bool ExactArc<PS>::has_endpoint(const IncidentCircle<PS>& i) const {
  return circle.is_same_intersection(src, i) || circle.is_same_intersection(dst, i);
}

} // namespace geode