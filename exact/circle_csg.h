// Robust constructive solid geometry for circular arc polygons in the plane
#pragma once

// These routines use algorithms almost identical to the actual polygon case, except using implicit
// representations for arcs and different (much higher order) predicates to check topology.

#include <other/core/exact/config.h>
#include <other/core/exact/quantize.h>
#include <other/core/array/Nested.h>

#include <vector>

namespace other {

// In floating point, we represent circular arcs by the two endpoints x0,x1 and q = 2*sagitta/|x1-x0|.
// q > 0 is counterclockwise, q < 0 is clockwise, q = 0 is a straight line.  In a circular arc polygon
// Array<CircleArc> a, arc k has endpoints a[k].x, a[k+1].x and q = a[k].q.
// arc k
struct CircleArc {
  Vector<real,2> x;
  real q;
};

// After quantization, we represent circles implicitly by center and radius, plus two boolean flags
// describing how to connect adjacent arcs.
struct ExactCircleArc {
  Vector<Quantized,2> center;
  Quantized radius;
  int index; // Index into the symbolic perturbation
  bool positive; // True if the arc is traversed counterclockwise
  bool left; // True if we use the intersection between this arc and the next to the left of the segment joining their centers
};

// Tweak quantized circles so that they intersect.
void tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs);
void tweak_arcs_to_intersect(Nested<ExactCircleArc>& arcs);

// Resolve all intersections between circular arc polygons, and extract the contour with given 
// Depth starts at 0 at infinity, and increases by 1 when crossing a contour from outside to inside.
// For example, depth = 0 corresponds to polygon_union.
OTHER_CORE_EXPORT Nested<CircleArc> split_circle_arcs(Nested<const CircleArc> arcs, const int depth);
OTHER_CORE_EXPORT Nested<ExactCircleArc> exact_split_circle_arcs(Nested<const ExactCircleArc> arcs, const int depth);

// The union of possibly intersecting circular arc polygons, assuming consistent ordering
template<class... Arcs> static inline Nested<Vec2> circle_arc_union(const Arcs&... arcs) {
  return split_circle_arcs(concatenate(arcs...),0);
}

// The intersection of possibly intersecting circular arc polygons, assuming consistent ordering.
template<class... Arcs> static inline Nested<Vec2> circle_arc_intersection(const Arcs&... arcs) {
  return split_circle_arcs(concatenate(arcs...),sizeof...(Arcs)-1);
}

// Signed area of circular arc polygons
OTHER_CORE_EXPORT real circle_arc_area(RawArray<const CircleArc> arcs);
OTHER_CORE_EXPORT real circle_arc_area(Nested<const CircleArc> arcs);

// Quantize from approximate to exact representations, taking care to ensure validity of the quantized result.
// If min_bounds isn't empty the Quantizer will use an appropriate scale to work with other features inside of min_bounds
OTHER_CORE_EXPORT Tuple<Quantizer<real,2>,Nested<ExactCircleArc>> quantize_circle_arcs(Nested<const CircleArc> arcs, const Box<Vector<real,2>> min_bounds=Box<Vector<real,2>>::empty_box());
OTHER_CORE_EXPORT Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2> quant, Nested<const ExactCircleArc> input);

OTHER_CORE_EXPORT Box<Vector<real,2>> approximate_bounding_box(const Nested<const CircleArc>& input);
OTHER_CORE_EXPORT ostream& operator<<(ostream& output, const CircleArc& arc);
OTHER_CORE_EXPORT ostream& operator<<(ostream& output, const ExactCircleArc& arc);

}
