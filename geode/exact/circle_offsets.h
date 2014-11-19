#pragma once
#include <geode/exact/circle_csg.h>
namespace geode {

// offset_arcs performs a csg_union on arcs then grows or shrinks (based on sign of d) interior by d
// Warning: Repeatedly feeding result of this function back to the input is strongly discouraged. Although geometric complexity ought
// to decrease or remain constant, rounding errors will frequently result in multiple arcs along slightly different circles in cases
// that should have resulted in a single arc. The exponential increase in number of arcs quickly cripples performance.
// * Calling offset_arcs on the original input with different offsets is one workaround
// * offset_shells preserves a higher precision representation that should avoid this issue
// * If user has a way to simplify arc contours (not currently available in geode) between iterations this would avoid the issue
Nested<CircleArc> offset_arcs(const Nested<const CircleArc> arcs, const real d);

// Generate closed contours around area covered by a disk of radius d moving along open arcs
Nested<CircleArc> offset_open_arcs(const Nested<const CircleArc> arcs, const real d);

// Repeatedly offset closed arcs by d
// If max_shells == -1 d must be negative and this will continue to offset arcs inward until result is empty
// If max_shells == -1 and d is zero or positive this will grind until it runs out of memory or otherwise do something horrible
vector<Nested<CircleArc>> offset_shells(const Nested<const CircleArc> arcs, const real d, const int max_shells = -1);

} // namespace geode
