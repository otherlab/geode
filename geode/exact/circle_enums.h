#pragma once
#include <geode/mesh/ids.h>
#include <geode/exact/circle_objects.h>

namespace geode {

// Circle CSG can involve a lot of tricky corner cases (although simulation of simplicity avoids a lot of these)
// Some enums are provided to help document intention in code and catch a few errors at compile time
// We also provide an assortment of small functions to pack and convert these enums for common cases

// Track if an arc procedes CCW around it's circle or CW
// For mesh edges that follow an arc around a circle, we assume the 'forward' halfedge travels CCW and the 'reverse' halfedge travels CW
enum class ArcDirection : bool { CCW = false, CW = true }; // directed_edge assumes these values are matched to behavior in HalfedgeGraph

// These make it easier to use ArcDirection as a winding
inline int sign(const ArcDirection d) { return (d == ArcDirection::CCW) ? 1 : -1; }
inline ArcDirection operator-(const ArcDirection dir) { return static_cast<ArcDirection>(dir==ArcDirection::CW); }
constexpr int direction_index(const ArcDirection dir) { return static_cast<int>(dir != ArcDirection::CCW); }

// CircleFace encodes if something is inside or outside of a circle
// We use this to determine if a path following just inside/outside a circle would cross an incident edges that ends on the circle
enum class CircleFace : bool { interior = false, exterior = true };

// We create a new id type for circles
GEODE_DEFINE_ID(CircleId)
// IncidentId combines a VertexId with a ReferenceSide to refer to an intersection point relative to a specific reference circle
GEODE_DEFINE_ID(IncidentId)

// This defines order of the two incident ids that refer to the same vertex
constexpr ReferenceSide first_iid_side = ReferenceSide::cl;
constexpr int incident_index(const ReferenceSide side) { return static_cast<int>(side != first_iid_side); }
static_assert(incident_index(first_iid_side) == 0, "first_iid_side doesn't match implementation of incident_id");

// Pack VertexId and ReferenceSide into an IncidentId
inline IncidentId incident_id(const VertexId vid, const ReferenceSide side) { assert(vid.valid()); return IncidentId(vid.idx()<<1 | incident_index(side)); }

// Convenience functions to manipulate VertexId/IncidentId
inline IncidentId iid_cl(const VertexId vid)     { return incident_id(vid, ReferenceSide::cl); }
inline IncidentId iid_cr(const VertexId vid)     { return incident_id(vid, ReferenceSide::cr); }
inline IncidentId opposite(const IncidentId iid) { assert(iid.valid()); return IncidentId(iid.idx() ^ 1); }
inline ReferenceSide side(const IncidentId iid)  { assert(iid.valid()); return static_cast<ReferenceSide>(iid.idx() & 1); }
inline VertexId   to_vid(const IncidentId iid)   { assert(iid.valid()); return VertexId(iid.idx() >> 1); }
inline bool is_same_vid(const IncidentId iid0, const IncidentId iid1) { assert(iid0.valid() && iid1.valid()); return (iid0.idx() | 1) == (iid1.idx() | 1); }
inline bool cl_is_reference(const IncidentId iid) { assert(iid.valid()); return cl_is_reference(side(iid)); }

// For an ExactArcGraph we use forward halfedges as CCW arcs and reversed halfedges as CW arcs
inline HalfedgeId directed_edge(const EdgeId eid, const ArcDirection direction) { assert(eid.valid()); return HalfedgeId(eid.idx()<<1 | direction_index(direction)); }
inline ArcDirection arc_direction(const HalfedgeId hid) { assert(hid.valid()); return static_cast<ArcDirection>(hid.idx() & 1); }
inline HalfedgeId ccw_edge(const EdgeId eid) { return directed_edge(eid, ArcDirection::CCW); }
inline HalfedgeId cw_edge (const EdgeId eid) { return directed_edge(eid, ArcDirection::CW ); }

// The interior face of a circle is adjacent to the ccw/forward halfedges
constexpr ArcDirection interior_edge_dir = ArcDirection::CCW;
// The exterior face of a circle is adjacent to the cw/reversed halfedges
constexpr ArcDirection exterior_edge_dir = ArcDirection::CW;

static inline ostream& operator<<(std::ostream& os, const ArcDirection d) {
  switch(d) {
    case ArcDirection::CCW: return os << "ArcDirection::CCW";
    case ArcDirection::CW: return os << "ArcDirection::CW";
  }
}
static inline ostream& operator<<(std::ostream& os, const CircleFace f) {
  switch(f) {
    case CircleFace::interior: return os << "CircleFace::interior";
    case CircleFace::exterior: return os << "CircleFace::exterior";
  }
}

} // namespace geode