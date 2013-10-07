#pragma once

#include <other/core/array/Array.h>
#include <other/core/array/Nested.h>
#include <other/core/exact/circle_csg.h>

namespace other {

// Apply a signed offset to an appropriately oriented set of circular arc polygons
OTHER_CORE_EXPORT Nested<CircleArc> offset_arcs(const Nested<const CircleArc> raw_arcs, const real offset_amount);
OTHER_CORE_EXPORT Nested<ExactCircleArc> exact_offset_arcs(const Nested<const ExactCircleArc> nested_arcs, const ExactInt offset_amount);

OTHER_CORE_EXPORT std::vector<Nested<CircleArc>> offset_shells(const Nested<const CircleArc> raw_arcs, const real shell_thickness, const int num_shells);

} // namespace other