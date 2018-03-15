#pragma once
#include <geode/array/Nested.h>
#include <geode/vector/Vector.h>
#include <geode/exact/circle_csg.h>

namespace geode {

// Greedily join closest endpoints together in a best-effort attempt to convert a set of open polylines to a set of closed contours
Nested<Vec2> join_fragments(const Nested<Vec2>& fragments);
Nested<CircleArc> join_fragments(const Nested<CircleArc>& fragments);

} // namespace geode
