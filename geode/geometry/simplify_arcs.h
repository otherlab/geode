#include <geode/exact/circle_csg.h>

namespace geode {

// This will collapse arc segments if it can ensure that no point moves by more than max_allowed_change.
// It currently doesn't try to refit q values or combine arcs that are nearly 'co-circular'
Array<CircleArc> simplify_arcs(const RawArray<const CircleArc> input, const real max_allowed_change, const bool is_closed);
Nested<CircleArc> simplify_arcs(const Nested<const CircleArc> input, const real max_point_movement, const bool is_closed=false);

} // namespace geode
