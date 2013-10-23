#pragma once

#include <geode/array/Array.h>
#include <geode/array/Nested.h>
#include <geode/exact/circle_csg.h>

namespace geode {

// Apply a signed offset to an appropriately oriented set of circular arc polygons
GEODE_CORE_EXPORT Nested<CircleArc> offset_arcs(const Nested<const CircleArc> raw_arcs, const real offset_amount);
GEODE_CORE_EXPORT Nested<ExactCircleArc> exact_offset_arcs(const Nested<const ExactCircleArc> nested_arcs, const ExactInt offset_amount);

GEODE_CORE_EXPORT std::vector<Nested<CircleArc>> offset_shells(const Nested<const CircleArc> raw_arcs, const real shell_thickness, const int num_shells);

} // namespace geode