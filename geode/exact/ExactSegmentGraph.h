#pragma once
#include <geode/array/NestedField.h>
#include <geode/exact/config.h>
#include <geode/geometry/BoxTree.h>
#include <geode/mesh/HalfedgeGraph.h>

namespace geode {

GEODE_DEFINE_ID(SegmentId)

struct ExactSegmentSet {
  const Field<const exact::Vec2, SegmentId> src_pts;
  const Field<const SegmentId, SegmentId> next;
  const Ref<BoxTree<exact::Vec2>> tree;

  int size() const { assert(src_pts.size() == next.size()); return next.size(); }

  exact::Perturbed2 src(const SegmentId s) const { return exact::Perturbed2(s.idx(), src_pts[s]); }
  exact::Perturbed2 dst(const SegmentId s) const { return src(next[s]); }

  bool segments_intersect(const SegmentId s0, const SegmentId s1) const;
  bool directions_oriented(const SegmentId s0, const SegmentId s1) const;
  exact::Vec2 approx_intersection(const SegmentId s0, const SegmentId s1) const;

  Array<Vector<SegmentId, 2>> intersection_pairs() const;
  uint8_t quadrant(const SegmentId s) const;
  explicit ExactSegmentSet(const Nested<const exact::Vec2> polys);
};

class ExactSegmentGraph {
 public:
  struct VertexInfo {
    Vector<SegmentId,2> segs;
    exact::Vec2 approx;
  };
  struct EdgeInfo {
    SegmentId segment;
  };

  ExactSegmentSet segs;
  Field<VertexInfo, VertexId> vertices;
  Field<EdgeInfo, EdgeId> edges;
  NestedField<Tuple<SegmentId,VertexId>,SegmentId> segment_verts;
  Ref<HalfedgeGraph> topology;
  explicit ExactSegmentGraph(const Nested<const exact::Vec2> polys);

  FaceId boundary_face() const;
 private:
  VertexId dst_id(const SegmentId s) const { return VertexId(segs.next[s].idx()); }
  EdgeId first_edge(const SegmentId s) const;
  void add_edge(const VertexId src, const VertexId dst, const SegmentId segment);

  EdgeId find_ray_hit(const exact::Perturbed2 ray_start, const SegmentId hit_segment) const;
  bool vertex_pt_upwards(const VertexId vid, const exact::Perturbed2 pt) const;
  HalfedgeId right_face_halfedge(const EdgeId eid) const;
  Array<HalfedgeId> path_from_infinity(const SegmentId hit_segment) const;
};

} // namespace geode