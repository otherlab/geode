// Robust constructive solid geometry for circular arc polygons in the plane
#pragma once

// These routines use algorithms almost identical to the actual polygon case, except using implicit
// representations for arcs and different (much higher order) predicates to check topology.

#include <geode/exact/config.h>
#include <geode/exact/quantize.h>

namespace geode {

// Returns a center and radius for an exact circle that passes within constructed_arc_endpoint_error_bound() units of each quantized vertex and has approxamently the correct curvature
GEODE_CORE_EXPORT Tuple<Vector<Quantized,2>, Quantized> construct_circle_center_and_radius(const Vector<Quantized, 2> x0, const Vector<Quantized, 2> x1, const real q);
GEODE_CORE_EXPORT Quantized constructed_arc_endpoint_error_bound();

// Returns a quantizer padded to be able to approximate straight lines
GEODE_CORE_EXPORT Quantizer<real,2> make_arc_quantizer(const Box<Vector<real,2>> arc_bounds);

} // namespace geode
