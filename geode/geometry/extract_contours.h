#pragma once
#include <geode/array/Nested.h>

namespace geode {

// Read in raster data and convert to closed contours interpolating between samples
// contour_edge_threshold: boundary is assumed to be at this value (i.g. 0.5 to a extract contour if input range is [0,1], or 0 if input range is [-1,1])
// default_sample_value: value used for border around 'samples'
// This doesn't attempt to do any simplification on the result. Most users will want to pass results to fit_polyarcs (from geometry/arc_fitting.h) or a polygon simplification routine. (I don't know why it isn't exposed in the header, but there's a polygon_simplify in polygon.cpp)
Nested<Vec2> extract_contours(const RawArray<const real, 2> samples, const real contour_edge_threshold, const real default_sample_value);

} // namespace geode
