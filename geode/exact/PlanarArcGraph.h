#pragma once
#include <geode/array/NestedField.h>
#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_enums.h>
#include <geode/exact/circle_objects.h>
#include <geode/utility/IdSet.h>
#include <geode/exact/quantize.h>
#include <geode/geometry/BoxTree.h>
#include <geode/geometry/traverse.h>
#include <geode/mesh/HalfedgeGraph.h>
#include <geode/mesh/ids.h>

namespace geode {

// A note on Unsigned vs Signed arcs:
// If you draw a circle and mark two points on the circle there are two possible arcs that connect them. We resolve this ambiguity differently depending on context.
// For many operations it is sufficient to treat an arc as a set of points (allowing us to check if two arcs intersect or not for example) without a specific direction.
// For these we order the endpoints so that the arc extends CCW from 'src' until it reaches 'dst'. For these we use UnsignedArcInfo (if we are working with ids) or ExactArc
// For other operations we need to describe a path along a circle that travels in a specific direction for which we use SignedArcInfo which includes a direction (CCW or CW),
// Signed arcs starts at a 'head', and extend CCW or CW to end at a 'tail'
// Signed arcs help us track windings and can be chained together into a continuous contour where the next arc starts at the end of the current arc

// This represents an arc as a subset of the points on a circle without any particular order
// As described above, points CCW from 'src' before 'dst' are included in the arc
struct UnsignedArcInfo {
  IncidentId src;
  IncidentId dst;
};

// This represents an arc that travels in a specific direction starting at 'head' and ending at 'tail'
struct SignedArcInfo {
  IncidentId _head;
  VertexId _tail;
  ArcDirection _direction;
  IncidentId head() const { return _head; }
  VertexId tail() const { return _tail; }
  ArcDirection direction() const { return _direction; }
  bool is_full_circle() const { return to_vid(_head) == _tail; }
  bool positive() const { return direction() == ArcDirection::CCW; }
};
static inline std::ostream& operator<<(std::ostream& os, const SignedArcInfo& x) {
  return os << "{" << x.head() << ", " << x.tail() << ", " << x.direction() << "}";
}

// To can store a sequence of signed arcs by saving start and direction, with tail implicitly specified by the start of the next arc
struct SignedArcHead {
  IncidentId iid; // Arc starts at this vertex traveling along the reference circle
  ArcDirection direction; // Arc travels along reference circle in this direction
  // With some packing and unpacking we could fit the reference circle id into padding and allow us to get tail_iid directly from the next head
  // This would probably be much better for cache coherence but benchmarking would be needed to tell if this outweighs the packing/unpacking cost
  // SignedCircleId ref_cid; // Would need to add a new data type similar to IncidentId/HalfedgeId that stores a CircleId and a direction flag
  // ArcDirection direction() const { return arc_direction(ref_cid); }
  // IncidentId tail_iid(const SignedArcHead next_head) const { return is_same_circle(ref_cid, next_head.ref_cid) ? next_head.iid : opposite(next_head.iid); }
};
static inline std::ostream& operator<<(std::ostream& os, const SignedArcHead& x) {
  return os << "{" << x.iid << ", " << x.direction << "}";
}

// Adaptor to make array of SignedArcHead behave like an array of SignedArcInfo
struct RawArcContour {
  RawArray<const SignedArcHead> heads; // Direct access to heads should be avoided in case future optimizations alter how closed/open contours are represented
  struct ArcIterator {
    SignedArcHead const* i;
    inline SignedArcInfo operator*() const { return { i->iid, to_vid((i+1)->iid), i->direction }; }
    void operator++() { ++i; }
    bool operator!=(const ArcIterator& rhs) const { return i != rhs.i; }
  };
  int n_arcs() const { return heads.size() - 1; }
  SignedArcInfo arc(const int i) const { return { heads[i].iid, to_vid(heads[i+1].iid), heads[i].direction }; }
  ArcIterator begin() const { return { heads.begin() }; }
  ArcIterator end() const { assert(!heads.empty()); return { heads.end() - 1 }; }
  SignedArcHead head() const { return heads.front(); }
  VertexId tail() const { assert(!heads.empty()); return to_vid(heads.back().iid); }
  SignedArcInfo back() const { return arc(heads.size() - 2); }
  bool is_closed() const;
};

// Represents a collection of contours
struct ArcContours {
  Nested<SignedArcHead, false> store; // Direct access to store should be avoided since future optimizations are likely to alter how closed/open contours are represented
  int size() const { return store.size(); }
  RawArcContour operator[](const int i) const { assert(!store[i].empty()); return { store[i] }; }
  ArrayIter<ArcContours> begin() const { return ArrayIter<ArcContours>(*this,0); }
  ArrayIter<ArcContours> end() const { return ArrayIter<ArcContours>(*this,size()); }
  RawArcContour back() const { return { store.back() }; }

  // Begin a new contour. Caller must also call end_open/closed_contour for each started contour
  void start_contour() { store.append_empty(); }
  // Append a new arc to the current contour from the specified intersection and direction
  // Added arc will end at the start of the next arc if any, the vertex passed to end_open_contour if used, or the start
  void append_to_back(const SignedArcHead arc) { store.append_to_back(arc); }
  void end_open_contour(const VertexId final_vertex); // Set the end of the current arc and mark the contour as open. Should be called at most once per started contour
  void end_closed_contour(); // Mark that the current arc ends at the start of the first arc (this should only be used if first arc actually starts somewhere on the current arc's circle)

  // Create a new closed contour with the given heads
  template<class TArray> void append_and_close(const TArray& array) {
    assert(!array.empty());
    store.append(array);
    end_closed_contour();
  }
  // Create a new open contour with the given heads and final endpoint
  template<class TArray> void append_open(const TArray& array, const VertexId final_vertex) { 
    store.append(array);
    end_open_contour(final_vertex);
  }
};

// Stores a set of ExactCircles (with no duplicates) and provides mapping from CircleIds to ExactCircles
template<Pb PS> class CircleSet {
 protected:
  IdSet<ExactCircle<PS>,CircleId> circles;
 public:
  int n_circles() const { return circles.size(); }
  Range<IdIter<CircleId>> circle_ids() const { return id_range<CircleId>(n_circles()); }
  const ExactCircle<PS>& circle(const CircleId cid) const { return circles[cid]; }
  CircleId find_cid(const ExactCircle<PS>& c) const { return circles.find(c); }
  CircleId get_or_insert(const ExactCircle<PS>& c) { return circles.get_or_insert(c); }
};

// A wrapper that stores an array of intersections and manages conversion between ids and geometric primitives
template<Pb PS> class VertexField {
 protected:
  // We store pairs of IncidentCircles instead of CircleIntersections so that we can return references for the majority of uses
  // Size should always be even since we only add IncidentCircles in pairs
  Field<IncidentCircle<PS>,IncidentId> incidents_;
 public:
  int n_vertices() const { assert((incidents_.size() % 2) == 0); return incidents_.size()/2; }
  int n_incidents() const { return incidents_.size(); }
  Range<IdIter<VertexId>> vertex_ids() const { return id_range<VertexId>(n_vertices()); }
  Range<IdIter<IncidentId>> incident_ids() const { return id_range<IncidentId>(n_incidents()); }

  const ApproxIntersection& approx(const IncidentId iid) const { return incidents_[iid].approx; }
  const ExactCircle<PS>& reference(const IncidentId iid) const { return incidents_[opposite(iid)].as_circle(); }
  const IncidentCircle<PS>& incident(const IncidentId iid) const { return incidents_[iid]; }

  const ExactCircle<PS>& cl(const VertexId vid) const { return incidents_[iid_cl(vid)].as_circle(); }
  const ExactCircle<PS>& cr(const VertexId vid) const { return incidents_[iid_cr(vid)].as_circle(); }

  ExactArc<PS> arc(const IncidentId src, const IncidentId dst) const {
    assert(is_same_circle(reference(src), reference(dst)));
    return {reference(src), incident(src), incident(dst)};
  }

  ExactArc<PS> arc(const UnsignedArcInfo info) const { return arc(info.src, info.dst); }

  // Adds an intersection. Caller must ensure this intersection does not already exist
  IncidentId append_unique(const ExactCircle<PS>& ref, const IncidentCircle<PS>& inc); // Adds both views of the vertex
  void extend_unique(const RawArray<const IncidentCircle<PS>> new_incidents);
};

template<Pb PS> class VertexSort; // Forward declaration for use by VertexSet

// Maintains a collection of circles and intersections with unique ids for all geometric primitives
template<Pb PS> class VertexSet : public CircleSet<PS> , public VertexField<PS> {
 protected:
  // Provide fast mapping between IncidentCircles in VertexField and ExactCircles in the CircleSet
  // Provides result of calling find_cid(incident(iid).as_circle()) without any expensive hashing
  Field<CircleId, IncidentId> iid_to_cid; 
  Hashtable<Vector<CircleId,2>,VertexId> vid_cache; // Used to avoid duplicates of the same intersection
 public:
  using VertexField<PS>::reference;
  using VertexField<PS>::incident;
  using CircleSet<PS>::get_or_insert;
  using CircleSet<PS>::find_cid;

  // These provide fast mapping from IncidentId to the CircleIds at that intersection
  CircleId reference_cid(const IncidentId iid) const;
  CircleId incident_cid(const IncidentId iid) const;
  // Returns the ordered pair of circles that define an intersection
  Vector<CircleId,2> circle_ids(const VertexId vid) const { return vec(reference_cid(iid_cl(vid)),reference_cid(iid_cr(vid))); }

  // Instead of handling necessary circles internally we force caller to add and track circles beforehand so that we avoid redundant lookups
  IncidentId get_or_insert(const IncidentCircle<PS>& inc, const CircleId ref_cid, const CircleId inc_cid);
  // Returns an invalid id if intersection wasn't previously added
  IncidentId try_find(const CircleId ref_cid, const CircleId inc_cid, const ReferenceSide side) const;
  // Choose reference side of a vertex to match a given circle
  // Caller is responsible for ensuring that vertex is on the given circle
  IncidentId find_iid(const VertexId vid, const CircleId ref_cid) const {
    const auto result = (reference_cid(iid_cl(vid)) == ref_cid) ? iid_cl(vid) : iid_cr(vid);
    assert(reference_cid(result) == ref_cid);
    return result;
  }

  // These functions constructs exact arc contours to approximate closed inexact contours
  // Constructed circles and intersections are added to the VertexSet with paths along those returned via ArcContours 
  // All vertices on the constructed path should be within a maximum 'error distance' (in the quantized space) of their corresponding vertex on the input path
  // Vertices closer than 'error distance' may be collapsed into a single vertex, and small 'helper arcs' can be added
  // Rather than trying to guarantee the 'error distance' is at the limits of the quantization resolution, we allow a few small internal approximations
  // A precise maximum value for the 'error distance' is hard to work out and may change, but it is likely around 10 quantized units.
  // Endpoint errors can be magnified by the q value for the middle of an arc. Contours with q larger than +/-1 can have worst case errors up to max(abs(q))*'error distance'
  // Approximations can result in the addition of small self intersections
  // Since arcs can be inserted or culled, current implementation doesn't guarantee any particular mapping between input arcs and the resulting contours
  void quantize_circle_arcs(const Quantizer<real,2>& quant, const RawArray<const CircleArc> src_arcs, ArcContours& result);
  ArcContours quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> src_arcs);
  // TODO: Add a version of quantize_circle_arcs for open inputs

  // Converts exact contours (which currently must be closed although open contours should be handled in the future) into CircleArcs
  // This function attempts to cull artifacts introduced by quantization, but will also simplify nearly degenerate arcs or contours
  Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2>& quant, const ArcContours& contours) const;

  // This combines arcs that continue along the same circle into a single arc wherever possible
  // The order of arcs in a closed contour can be rotated if last and first arcs are merged, this will change the 'start' of the contour
  // This won't happen for open contours even if they happen to start and end at the same point
  ArcContours combine_concentric_arcs(const ArcContours& contours, const VertexSort<PS>& incident_order) const;

  // Utility operators for SignedArcInfo
  CircleId reference_cid(const SignedArcInfo arc) const { return reference_cid(arc.head()); }
  IncidentId tail_iid(const SignedArcInfo arc) const { return find_iid(arc.tail(), reference_cid(arc)); }
  // This removes the direction from a SignedArcInfo to get endpoints in CCW order  
  UnsignedArcInfo ccw_arc(const SignedArcInfo arc) const {
    return (arc.direction() == ArcDirection::CCW) ? UnsignedArcInfo({ arc.head(), tail_iid(arc) })
                                                  : UnsignedArcInfo({ tail_iid(arc), arc.head() });
  }
};

// This stores a bounding box hierarchy for a set of circles clipped to regions that have arcs
// (i.e. Each circle uses the combined bounding box of all arcs on that circle)
// Since we use the circles as leaves arcs can be split into multiple arcs (as long as combined bounding box is the same) without needing to recompute tree
// When we have multiple arcs on the same circle (such as after splitting) grouping by circles ensures pairs of leaf nodes always refer to unique intersection points
// Before splitting there will usually only be a single arc on each circle making this just as efficient as a bounding box hierarchy on the arcs
// After splitting we only need a few quick raycasts which can be mapped to the correct arcs using VertexSort
template<Pb PS> class CircleTree {
 public:
  Ref<const BoxTree<Vec2>> tree; // The raw BoxTree

  // This constructor will computes necessary bounding box around any parts of circles that contain arcs
  CircleTree(const VertexSet<PS>& verts, const ArcContours& contours);

  // We don't provide a default constructor, but allow deferred initialization borrowing 'uninit' semantics from Array
  // Caller is responsible for assigning a properly initialized CircleTree to this instance before trying to access
  CircleTree(Uninit);

  // Extract a CircleId for a given node (which must be a leaf)
  CircleId prim(const int n) const;

  // Traverse the box tree and find any circles that have arcs inside 'bounds'
  // Warning: This is not all circles that intersect bounds! Circles are clipped to 'active' regions traversed by contours in initialization
  Array<CircleId> circles_active_near(const CircleSet<PS>& circles, const Box<Vec2> bounds) const;
};

// VertexSort iterator that traverses IncidentIds in CCW order around a circle
struct IncidentCirculator {
  RawArray<const IncidentId> incidents;
  int i;
  IncidentCirculator &operator++() { ++i; if(i == incidents.size()) i = 0; return *this; }
  IncidentCirculator &operator--() { if(i == 0) i = incidents.size(); --i; return *this; }
  IncidentId operator*() const { return incidents[i]; }
  IncidentId prev() const { return (i == 0) ? incidents.back() : incidents[i-1]; }
  IncidentId next() const { return (i + 1 == incidents.size()) ? incidents.front() : incidents[i+1]; }
};

// Find permutation of intersections to replace various geometric primitives with simple integer comparisons
// Doesn't support mutation of vertices after construction
template<Pb PS> class VertexSort {
 protected:
  NestedField<IncidentId, CircleId> circle_permutation; // Each IncidentId sorted in ccw order around its reference circle
  Field<int, IncidentId> incident_permutation; // Index into circle_permutation.raw.flat for each IncidentId
 public:
  VertexSort(Uninit); // Caller is responsible for assigning a properly initialized VertexSort before other things will work
  VertexSort(const VertexSet<PS>& vertices);

  // All intersections on a given circle
  RawArray<const IncidentId> circle_incidents(const CircleId cid) const { return circle_permutation[cid]; }

  // Step forward by a single intersection
  IncidentId ccw_next(const CircleId cid, const IncidentId iid) const {
    const int offset = incident_permutation[iid];
    assert(circle_permutation.raw.flat[offset] == iid); // Check that offset points back to iid
    assert(circle_permutation.offset_range(cid).contains(offset)); // Check that we are looking at the right circle
    const int next_offset = (offset == circle_permutation.back_offset(cid)) ? circle_permutation.front_offset(cid) : offset+1;
    return circle_permutation.raw.flat[next_offset];
  }

  // Step back by a single intersection
  IncidentId ccw_prev(const CircleId cid, const IncidentId iid) const {
    const int offset = incident_permutation[iid];
    assert(circle_permutation.raw.flat[offset] == iid); // Check that offset points back to iid
    assert(circle_permutation.offset_range(cid).contains(offset)); // Check that we are looking at the right circle
    const int prev_offset = (offset == circle_permutation.front_offset(cid)) ? circle_permutation.back_offset(cid) : offset-1;
    return circle_permutation.raw.flat[prev_offset];
  }

  // Iterator that can walk around a circle
  IncidentCirculator circulator(const CircleId cid, const IncidentId iid) const {
    assert(circle_permutation.offset_range(cid).contains(incident_permutation[iid]));
    return IncidentCirculator({circle_permutation[cid], incident_permutation[iid] - circle_permutation.front_offset(cid)});
  }

  // Can replace c.intersections_sorted(i0,i1) with (psudo_angle(iid0) < psudo_angle(iid1))
  inline int psudo_angle(const IncidentId iid) const;

  // As ExactArc::interior_contains, but should be significantly faster
  bool arc_interior_contains(const UnsignedArcInfo a, const IncidentId i) const;
};

// Stores a set of arcs and their planar embedding
template<Pb PS> class PlanarArcGraph : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_EXPORT)

  // This are exposed for external access, but should generally be treated as read-only
  CircleTree<PS> circle_tree;
  VertexSet<PS> vertices;
  VertexSort<PS> incident_order;
  Field<int, EdgeId> edge_windings;
  Field<EdgeId, IncidentId> outgoing_edges;
  Field<IncidentId, EdgeId> edge_srcs;
  Ref<HalfedgeGraph> topology;

protected:
  // To construct a PlanarArcGraph, but doesn't initialize members
  // User can safely access vertices by reference to add circles and intersections
  // Must then call embed_arcs which will initialize remaining members
  PlanarArcGraph(Uninit);

  // This constructor splits edges and computes the embedding
  PlanarArcGraph(const VertexSet<PS>& _vertices, const ArcContours& contours);
public:
  // Initializes a PlanarArcGraph for the given contours
  // All vertices and circles must have been added to vertices
  void embed_arcs(const ArcContours& contours);

  inline CircleId circle_id(const EdgeId eid) const;
  inline IncidentId src(const EdgeId eid) const;
  inline IncidentId dst(const EdgeId eid) const;
  inline ExactArc<PS> arc(const EdgeId eid) const;

  inline CircleId circle_id(const HalfedgeId eid) const;
  inline IncidentId src(const HalfedgeId hid) const;
  inline IncidentId dst(const HalfedgeId hid) const;

  // Find edge such that starts at or contains src. If no such edge exists, returns an invalid id
  EdgeId outgoing_edge(const IncidentId src) const;

  // Find incident such that vertices.arc(*result, result.next()).half_open_contains(i)
  IncidentCirculator find_prev(const CircleId cid, const IncidentCircle<PS>& i) const;
  // Find incident such that vertices.arc(*result, result.next()).contains_horizontal(i)
  IncidentCirculator find_prev(const CircleId cid, const IncidentHorizontal<PS>& i) const;

  // Find edge (if any) that contains i. If no such edge exists returns an invalid id
  EdgeId find_edge(const CircleId ref_cid, const IncidentHorizontal<PS>& i) const;

  // Perform raycast from right to left along i.line ending at i
  Array<HalfedgeId> leftward_raycast_to(const CircleId ref_cid, const IncidentHorizontal<PS>& i) const;

  // This is the 'outside' face of the graph. A point at infinity (or just really far away) will be inside this face
  FaceId boundary_face() const; // Warning, if topology has no faces (when graph has no geometry) this will still return a valid FaceId

  // This converts a connected chain of halfedges into ArcContours
  ArcContours edges_to_closed_contours(const Nested<const HalfedgeId> edges) const;

  // See VertexSet::combine_concentric_arcs for more details
  ArcContours combine_concentric_arcs(const ArcContours& contours) const { return vertices.combine_concentric_arcs(contours, incident_order); }

  // Similar to VertexSet::unquantize_circle_arcs, but accepts connected chains of halfedges and first calls combine_concentric_arcs  
  Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const HalfedgeId> contours) const;

  // Does the arc from a.x to a.y contain i?
  bool arc_interior_contains(const UnsignedArcInfo a, const IncidentId i) const;

protected:
  // Compute edges crossed on path starting at infinity that ends somewhere on target_eid
  // Last crossing will always be on target_eid
  // This is used internally to initialize topology faces, but after faces are constructed it is probably more efficient to traverse graph
  Array<HalfedgeId> path_from_infinity(const EdgeId target_eid) const;

  void init_borders_and_faces();
};

// This allows incremental construction of arcs for a PlanarArcGraph
template<Pb PS> struct ArcAccumulator {
  // Most usage will need to directly add primitives to these
  VertexSet<PS> vertices;
  ArcContours contours;

  // Shortcut to handle turning a circle into an arc (handles construction of placeholder intersection)
  void add_full_circle(const ExactCircle<PS>& c, const ArcDirection dir);

  // calls contours.append_to_back(h), but performs additional error checking for debug builds
  void append_to_back(const SignedArcHead h); 

  // Copy intersections out of a VertexSet and add contours with the new ids
  void copy_contours(const ArcContours& src_contours, const VertexSet<PS>& src_vertices);

  // Use vertices and contours to create a PlanarArcGraph
  Ref<PlanarArcGraph<PS>> compute_embedding() const;

  // Convenience function to avoid repeated code. Computes the embedding and returns contours for union
  Tuple<Ref<PlanarArcGraph<PS>>,Nested<HalfedgeId>> split_and_union() const;
};

template<Pb PS> Field<bool, FaceId> faces_greater_than(const PlanarArcGraph<PS>& g, const int depth);
template<Pb PS> Field<bool, FaceId> odd_faces(const PlanarArcGraph<PS>& g);

// Performs quantization of contours and computes planar embedding
template<Pb PS> Ref<PlanarArcGraph<PS>> quantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc> arcs);
// As above, but computes and returns an appropriate quantizer
template<Pb PS> Tuple<Quantizer<real,2>, Ref<PlanarArcGraph<PS>>> quantize_circle_arcs(const Nested<const CircleArc> arcs);

////////////////////////////////////////////////////////////////////////////////

template<Pb PS> inline CircleId PlanarArcGraph<PS>::circle_id(const EdgeId eid) const { return vertices.reference_cid(src(eid)); }
template<Pb PS> inline IncidentId PlanarArcGraph<PS>::src(const EdgeId eid) const { return edge_srcs[eid]; }
template<Pb PS> inline IncidentId PlanarArcGraph<PS>::dst(const EdgeId eid) const { return vertices.find_iid(topology->dst(eid), circle_id(eid)); }
template<Pb PS> inline ExactArc<PS> PlanarArcGraph<PS>::arc(const EdgeId eid) const { return vertices.arc(src(eid),dst(eid)); }

template<Pb PS> inline CircleId PlanarArcGraph<PS>::circle_id(const HalfedgeId hid) const { return circle_id(HalfedgeGraph::edge(hid)); }
template<Pb PS> inline IncidentId PlanarArcGraph<PS>::src(const HalfedgeId hid) const { return HalfedgeGraph::is_forward(hid) ? src(HalfedgeGraph::edge(hid)) : dst(HalfedgeGraph::edge(hid)); }
template<Pb PS> inline IncidentId PlanarArcGraph<PS>::dst(const HalfedgeId hid) const { return HalfedgeGraph::is_forward(hid) ? dst(HalfedgeGraph::edge(hid)) : src(HalfedgeGraph::edge(hid)); }

template<Pb PS> inline int VertexSort<PS>::psudo_angle(const IncidentId iid) const { return incident_permutation[iid]; }

} // namespace geode
