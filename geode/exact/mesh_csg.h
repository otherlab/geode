// Robust constructive solid geometry for triangle meshes
#pragma once

#include <geode/exact/config.h>
#include <geode/mesh/TriangleSoup.h>
namespace geode {

// If depth is this, faces at all depths are returned
const int all_depths = std::numeric_limits<int>::min();

// Resolve all intersections between triangle soups.
GEODE_CORE_EXPORT Tuple<Ref<const TriangleSoup>,Array<Vec3>>
split_soup(const TriangleSoup& faces, Array<const Vector<double,3>> X, const int depth);

GEODE_CORE_EXPORT Tuple<Ref<const TriangleSoup>,Array<Vec3>>
split_soup(const TriangleSoup& faces, Array<const Vector<double,3>> X, Array<const int> depth_weights, const int depth);

GEODE_CORE_EXPORT Tuple<Ref<const TriangleSoup>,Array<exact::Vec3>>
exact_split_soup(const TriangleSoup& faces, Array<const exact::Vec3> X, const int depth);

GEODE_CORE_EXPORT Tuple<Ref<const TriangleSoup>,Array<exact::Vec3>>
exact_split_soup(const TriangleSoup& faces, Array<const exact::Vec3> X, Array<const int> depth_weights, const int depth);

}
