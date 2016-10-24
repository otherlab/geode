#pragma once
#include <geode/exact/circle_csg.h>
namespace geode {

// Splits arcs into contiguous solid components and finds areas that are less than 'd' from two or more different components
// If d is negative, components will be the holes instead
GEODE_CORE_EXPORT Nested<CircleArc> find_overlapping_offsets(const Nested<const CircleArc> arcs, const real d);


} // namespace geode
