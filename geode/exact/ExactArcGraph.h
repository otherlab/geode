#pragma once
#include <geode/array/Field.h>
#include <geode/exact/circle_objects.h>
#include <geode/exact/quantize.h>
#include <geode/exact/circle_csg.h>
#include <geode/exact/VertexSet.h>
#include <geode/mesh/HalfedgeGraph.h>
#include <geode/mesh/ids.h>

namespace geode {

GEODE_DEFINE_ID(CircleId)

// IncidentId combines a VertexId with a ReferenceSide to refer to an intersection point relative to a specific reference circle
GEODE_DEFINE_ID(IncidentId)
inline IncidentId incident_id(const VertexId vid, const ReferenceSide side) { assert(vid.valid()); return IncidentId(vid.idx()<<1 | static_cast<bool>(side)); }
inline IncidentId iid_cl(const VertexId vid)     { return incident_id(vid, ReferenceSide::cl); }
inline IncidentId iid_cr(const VertexId vid)     { return incident_id(vid, ReferenceSide::cr); }
inline IncidentId reversed(const IncidentId iid) { assert(iid.valid()); return IncidentId(iid.idx() ^ 1); }
inline ReferenceSide side(const IncidentId iid)  { assert(iid.valid()); return static_cast<ReferenceSide>(iid.idx() & 1); }
inline VertexId   to_vid(const IncidentId iid)   { assert(iid.valid()); return VertexId(iid.idx() >> 1); }
inline bool is_same_vid(const IncidentId iid0, const IncidentId iid1) { assert(iid0.valid() && iid1.valid()); return (iid0.idx() | 1) == (iid1.idx() | 1); }
inline bool cl_is_reference(const IncidentId iid) { assert(iid.valid()); return cl_is_reference(side(iid)); }

template<Pb PS> class ExactArcGraph {
 public:
  struct EdgeValue {
    int weight; // Total number of times edge was added in any orientation
    int winding; // Total number of times ccw halfedge - cw halfedge was added
    EdgeValue() = default;
    EdgeValue(const int _weight, const int _winding) : weight(_weight), winding(_winding) { }
    void operator+=(const EdgeValue rhs) { weight += rhs.weight; winding += rhs.winding; }
  };

  struct EdgeInfo {
    EdgeValue value;
    ReferenceSide src_side;
    ReferenceSide dst_side;
  };

  // These should be safe to read, but modified at your own risk
  VertexSet<PS> vertices;
  Field<EdgeInfo, EdgeId> edges;
  Ref<HalfedgeGraph> graph; // this->split_edges() must be called before graph will get valid border/face data

  ExactArcGraph();
  ~ExactArcGraph();

  int n_vertices()  const { assert(vertices.size() == graph->n_vertices()); return vertices.size(); }
  int n_incidents() const { return 2*n_vertices(); }
  int n_edges()     const { assert(edges.size() == graph->n_edges()); return edges.size(); }

  Range<IdIter<VertexId>>   vertex_ids()   const { return id_range<VertexId>  (n_vertices()); }
  Range<IdIter<EdgeId>>     edge_ids()     const { return id_range<EdgeId>    (n_edges()); }
  Range<IdIter<IncidentId>> incident_ids() const { return id_range<IncidentId>(n_incidents()); }

  const ExactCircle<PS>& reference(const IncidentId iid) const { return vertices[to_vid(iid)].reference(side(iid)); }
  const IncidentCircle<PS> incident(const IncidentId iid) const { return vertices[to_vid(iid)].incident(side(iid)); }

  const IncidentId src(const EdgeId eid) const { return incident_id(graph->src(eid), edges[eid].src_side); }
  const IncidentId dst(const EdgeId eid) const { return incident_id(graph->dst(eid), edges[eid].dst_side); }

  const IncidentId src(const HalfedgeId hid) const { return HalfedgeGraph::is_forward(hid) ? src(HalfedgeGraph::edge(hid)) : dst(HalfedgeGraph::edge(hid)); }
  const IncidentId dst(const HalfedgeId hid) const { return HalfedgeGraph::is_forward(hid) ? dst(HalfedgeGraph::edge(hid)) : src(HalfedgeGraph::edge(hid)); }

  const ExactCircle<PS>& circle(const EdgeId eid) const { return reference(src(eid)); }
  const ExactArc<PS> arc(const EdgeId eid) const { return ExactArc<PS>({circle(eid), incident(src(eid)), incident(dst(eid)) } ); }

  EdgeId add_arc(const ExactArc<PS>& arc, const EdgeValue value);
  EdgeId add_full_circle(const ExactCircle<PS>& c, const EdgeValue value);

  // Converts inexact polyarc contours into exact contours and adds them to the graph
  // Result will have one contour per input arc, but these can be empty; For example, when an input contour was simplified to a single vertex
  // Doesn't compute planar embedding of any added geometry (split_edges must be called manually and will destroy coorespondence of returned contours)
  Nested<HalfedgeId> quantize_and_add_arcs(const Quantizer<real,2>& quant, const Nested<const CircleArc>& src_arcs);

  // For each contours, travels the graph and produces coresponding inexact polyarcs
  Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2>& quant, const Nested<const HalfedgeId> contours) const;

  // Combines all coincident edges, adds verticies at intersections, and computes a planar embedding
  // WARNING: Only verticies are preserved. This invalidates all EdgeIds as well as creating a new "graph" object
  void split_edges();
  FaceId boundary_face() const;

  Field<int, FaceId> face_winding_depths() const;
 protected:
  VertexId get_or_insert(const CircleIntersectionKey<PS>& i);
  VertexId get_or_insert_intersection(const CircleIntersection<PS>& i);

  Array<HalfedgeId> quantize_and_add_arcs(const Quantizer<real,2>& quant, const RawArray<const CircleArc> src_arcs, const EdgeId base_ref_id);
  Field<IncidentId, IncidentId> ccw_next() const; // Forward links to travel ccw around any circle
  void new_noncoincident_edges(); // This invalidates all edge ids
  EdgeId _unsafe_split_edge(const EdgeId e0, const IncidentId iid);
  Field<Box<exact::Vec2>, EdgeId> edge_boxes() const;
  Ref<const BoxTree<exact::Vec2>> get_edge_tree() const;
  void embed_vertex(const VertexId vid); // Requires no coincident edges at vertex
  void compute_borders_and_faces();
  Array<HalfedgeId> horizontal_raycast(const ExactHorizontal<PS>& y_ray, const BoxTree<exact::Vec2>& edge_tree) const;
  Array<HalfedgeId> path_from_infinity(const HalfedgeId seed_he, const BoxTree<exact::Vec2>& edge_tree) const;
  Array<HalfedgeId> path_to_infinity(const HalfedgeId seed_he, const BoxTree<exact::Vec2>& edge_tree) const;
  void compute_embedding();
};

// Use fill rules to select faces of an ExactArcGraph based on winding depth
template<Pb PS> Field<bool, FaceId> faces_greater_than(const ExactArcGraph<PS>& g, const int depth);
template<Pb PS> Field<bool, FaceId> odd_faces(const ExactArcGraph<PS>& g);

template<Pb PS> GEODE_CORE_EXPORT Box<Vec2> bounding_box(const ExactArcGraph<PS>& g);

template<Pb PS> GEODE_CORE_EXPORT Tuple<Quantizer<real,2>,ExactArcGraph<PS>> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Box<Vec2> min_bounds=Box<Vec2>::empty_box());
template<Pb PS> GEODE_CORE_EXPORT ExactArcGraph<PS> quantize_circle_arcs(const Nested<const CircleArc> arcs, const Quantizer<real,2> quant);

} // namespace geode
