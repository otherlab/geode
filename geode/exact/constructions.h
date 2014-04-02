// Nearly exact geometric constructions
#pragma once

#include <geode/exact/config.h>
#include <geode/vector/Vector.h>
namespace geode {

// Construct the intersection of two segments, assuming they actually do intersect.
// Due to an initial interval filter, the result differs from the true value by up to segment_segment_intersection_threshold.
GEODE_CORE_EXPORT GEODE_PURE exact::Vec2 segment_segment_intersection(const exact::Perturbed2 a0, const exact::Perturbed2 a1, const exact::Perturbed2 b0, const exact::Perturbed2 b1);
const int segment_segment_intersection_threshold = 100;

// For testing purposes
GEODE_CORE_EXPORT void construction_tests();

}
