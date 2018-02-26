#pragma once
#include <geode/array/Array.h>
#include <geode/exact/circle_csg.h>

namespace geode {

// Convert CircleArcs to polylines with at least enough samples to stay within max_deviation of original path
Array<Vec2> discretize_arcs(const RawArray<const CircleArc> arc_points, const bool closed, const real max_deviation);
Nested<Vec2> discretize_nested_arcs(const Nested<const CircleArc> arc_points, const bool closed, const real max_deviation);

// Find the 'q' value for an ArcSegment or CircleArc with the given endpoints that would go through a target point
// p0 is start of arc
// p1 is end of arc
// p3 is point on arc between start and end
real fit_q(const Vec2 p0, const Vec2 p1, const Vec2 p3);
// As fit_q, but returns the range of values that stay within the given error of p3
Box<real> fit_q_range(const Vec2 p0, const Vec2 p1, const Vec2 p3, const real allowed_error);

// Fit CircleArcs to polylines/polygons at a specified tolerance
// All endpoints will be within allowed_error of resulting arcs, error for interior of segments assumes endpoints were exactly on arc and thus can be slightly larger than allowed_error
Array<CircleArc> fit_arcs(const RawArray<const Vec2> poly, const real allowed_error, const bool closed);
Nested<CircleArc> fit_polyarcs(const Nested<const Vec2> polys, const real allowed_error, const bool closed);

} // geode namespace
