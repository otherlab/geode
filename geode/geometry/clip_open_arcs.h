#pragma once
#include <geode/exact/circle_csg.h>
namespace geode {

enum class ClippingMode { AllowRentry, // Breaks paths into multiple unconnected portions
                          AllOrNothing, // Remove all of path if any part of path would be clipped
                          OnlyStartAtStart, // 'Ray cast' along each path until it is clipped. Remainder of path is discarded even if it later reenters closed_arcs
                        };

// Walk along open_arcs returning only the portions that are inside the union of closed_arcs
Nested<CircleArc> arc_path_intersection(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode);

// Walk along open_arcs return only the portions that are outside the union of closed_arcs
Nested<CircleArc> arc_path_difference(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode);

// As above, but returning a tuple with .second containing a map indicating which open_arcs generated each result path
//   i.e. result.first[i] will be a subset of open_arcs[result.second[i]]
std::pair<Nested<CircleArc>,Array<int>> arc_path_intersection_with_coorespondence(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode);
std::pair<Nested<CircleArc>,Array<int>> arc_path_difference_with_coorespondence(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode);

} // geode namespace