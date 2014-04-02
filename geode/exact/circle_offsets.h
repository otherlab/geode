#pragma once

#include <geode/exact/circle_csg.h>
namespace geode {

// Perform a csg_union on arcs then grow or shrink (based on sign of d) interior by d
GEODE_CORE_EXPORT Nested<CircleArc> offset_arcs(const Nested<const CircleArc> arcs, const real d);

// Generate closed contours around area covered by a disk of radius d moving along open arcs
GEODE_CORE_EXPORT Nested<CircleArc> offset_open_arcs(const Nested<const CircleArc> arcs, const real d);

} // namespace geode
