#include "find_overlapping_offsets.h"

#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_offsets.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/exact_circle_offsets.h>
#include <geode/exact/PlanarArcGraph.h>
#include <geode/exact/scope.h>
#include <geode/structure/UnionFind.h>

namespace geode {

static constexpr Pb PS = Pb::Implicit;

// 'merge' adjacent faces into components. Returns a ComponentId for each face
static Field<ComponentId,FaceId> cluster_faces(const HalfedgeGraph& g, const RawField<const bool, FaceId> face_groups) {
  // Use a UnionFind to merge faces in the same group that share an edge
  auto parents = Array<int>{face_groups.size(), uninit};
  auto contiguous_clusters = RawUnionFind{parents};
  for(const EdgeId eid : g.edges()) {
    auto edge_faces = g.faces(eid);
    if(face_groups[edge_faces[0]] == face_groups[edge_faces[1]]) {
      contiguous_clusters.merge(edge_faces[0].idx(), edge_faces[1].idx());
    }
  }

  // Compact roots down to get consecutive integers for each component
  auto result = Field<ComponentId, FaceId>{face_groups.size()};
  auto next_unique_component = IdIter<ComponentId>{ComponentId{0}};
  for(const FaceId fid : face_groups.id_range()) {
    const auto root = FaceId{contiguous_clusters.find(fid.idx())};
    auto& root_component = result[root];
    if(!root_component.valid()) root_component = *(next_unique_component++);
    result[fid] = ComponentId{root_component};
  }

  return result;
}

template<class Fn> static void unordered_traverse_halfedges(const PlanarArcGraph<PS>& graph, const RawArcContour contour, Fn&& fn) {
  for(const SignedArcInfo signed_arc : contour) {
    const auto ccw_arc = graph.vertices.ccw_arc(signed_arc);
    const CircleId arc_circle = graph.vertices.reference_cid(ccw_arc.src);
    auto incident_iter = graph.incident_order.circulator(arc_circle, ccw_arc.src);
    do {
      const auto eid = graph.outgoing_edges[*incident_iter];
      assert(graph.topology->valid(eid));
      const auto hid = directed_edge(eid, signed_arc.direction());
      fn(hid);
      ++incident_iter;
    } while(*incident_iter != ccw_arc.dst);
  }
}

template<class Flag> static NestedField<Flag, HalfedgeId> flag_contour_halfedges(const PlanarArcGraph<PS>& graph, const ArcContours& contours, RawArray<const Flag> contour_flags) {
  assert(contours.size() == contour_flags.size());
  const int n_contours = contours.size();

  // Walk over each contour and count the number of references to each edge so that we can store result as a Nested
  auto flag_counts = Field<int, HalfedgeId>{graph.topology->n_halfedges()};
  auto increment_flag_count = [&flag_counts](const HalfedgeId hid) { ++flag_counts[hid]; };
  for(const auto contour : contours) {
    unordered_traverse_halfedges(graph, contour, increment_flag_count);
  }

  // Allocate result, then rewalk the contours saving the actual flags
  auto result = NestedField<Flag, HalfedgeId>{flag_counts, uninit};
  for(const int i : range(n_contours)) {
    const auto flag = contour_flags[i];
    unordered_traverse_halfedges(graph, contours[i], [&flag_counts, result, flag](const HalfedgeId hid) {
      result[hid][--flag_counts[hid]] = flag;
    });
  }
  assert(flag_counts.flat.contains_only(0));

  return result;
}

// Returns which faces of graph are enclosed by two or more unique components
// For each halfedge, edge_components should have list of components that would be entered when crossed (which will be same as those exited when crossing reversed halfedge)
// This assumes components were initially closed (i.e. that all paths through graph cross the same components for the same start/end faces)
static Field<bool, FaceId> find_overlapping_components(const HalfedgeGraph& graph, const NestedField<ComponentId,HalfedgeId>& edge_components, const FaceId boundary_face) {
  // This builds a tree of all faces with boundary face at the root and each child being each face reachable by one edge crossing
  // 
  // Then it traverses that tree depth first tracking which components it is currently inside
  struct OverlapFinder {
    const HalfedgeGraph& graph;
    const NestedField<ComponentId, HalfedgeId>& edge_components;

    NestedField<HalfedgeId, FaceId> crossing_tree;
    std::map<ComponentId,int> current_depths; // Possible optimization: Could replace this with a Field<int, ComponentId> and a count of non-zero elements. I'm not sure if that would be worthwhile though.
    Field<bool, FaceId> result;

    OverlapFinder(const HalfedgeGraph& new_graph, const NestedField<ComponentId, HalfedgeId>& new_edge_components, const FaceId boundary_face)
     : graph(new_graph)
     , edge_components(new_edge_components)
     , result(graph.n_faces(), uninit)
    {
      const Field<CrossingInfo, FaceId> crossing_info = get_crossing_depths(graph, boundary_face);

      auto num_child_crossings = Field<int,FaceId>{graph.n_faces()};
      for(const auto& crossing : crossing_info.flat) {
        const auto crossed_edge = crossing.next;
        if(!crossed_edge.valid()) {
          assert(&crossing == &crossing_info[boundary_face]);
          continue;
        }
        const auto parent_face = graph.opp_face(crossed_edge);
        ++num_child_crossings[parent_face];
      }

      crossing_tree = NestedField<HalfedgeId,FaceId>{num_child_crossings};

      for(const auto& crossing : crossing_info.flat) {
        const auto crossed_edge = crossing.next;
        if(!crossed_edge.valid()) {
          assert(&crossing == &crossing_info[boundary_face]);
          continue;
        }
        const auto parent_face = graph.opp_face(crossed_edge);
        crossing_tree[parent_face][--num_child_crossings[parent_face]] = crossed_edge;
      }

      assert(num_child_crossings.flat.contains_only(0));
      assert(crossing_tree.size() == graph.n_faces());
      if(graph.valid(boundary_face))
        traverse_crossing_tree(boundary_face);
    }

    void update_depths(const HalfedgeId crossed_edge) {
      assert(edge_components.valid(crossed_edge));
      for(const ComponentId entered : edge_components[crossed_edge]) current_depths[entered]++;
      for(const ComponentId exited : edge_components[HalfedgeGraph::reverse(crossed_edge)]) {
        auto iter = current_depths.find(exited);
        assert(iter != current_depths.end());
        --iter->second;
        if(iter->second == 0) current_depths.erase(iter);
      }
    }

    void traverse_crossing_tree(const FaceId current_face) {
      assert(std::all_of(current_depths.cbegin(), current_depths.cend(), [](const std::map<ComponentId,int>::value_type& v) {
        return (v.second >= 1);
      }));
      assert(result.valid(current_face));
      assert(crossing_tree.valid(current_face));
      result[current_face] = (current_depths.size() >= 2);
      for(const HalfedgeId child_crossing : crossing_tree[current_face]) {
        assert(graph.opp_face(child_crossing) == current_face);
        const auto child_face = graph.face(child_crossing);
        update_depths(child_crossing); // Update depths in place (O(log(n))) instead of copying (O(n))
        traverse_crossing_tree(child_face); // Depth first traverse the tree
        update_depths(graph.reverse(child_crossing)); // Since we updated depths in place, need to undo change
      }
    }
  };
  return OverlapFinder{graph, edge_components, boundary_face}.result;
}

#ifdef NDEBUG
#else
// Same as find_overlapping_components, but slower (in worst case)
static Field<bool, FaceId> find_overlapping_components_slow(const HalfedgeGraph& graph, const NestedField<ComponentId,HalfedgeId>& edge_components, const FaceId boundary_face) {
  // This version is something like O(n^2*log n) in number of faces
  // Average case should be much better and this is probably quite fast in most cases
  // However an unlucky traversal order on a graph with half concentric nested faces and remaining faces all at the lowest level would cause the worst case runtime
  const Field<CrossingInfo, FaceId> crossing_info = get_crossing_depths(graph, boundary_face);

  enum class FaceEnclosedBy : unsigned char { Unknown, Zero, One, TwoOrMore };
  auto face_info = Field<FaceEnclosedBy, FaceId>{graph.n_faces()};
  // These are outside of the loop so that they don't need to be reallocated for every face
  Array<HalfedgeId> path_to_infinity;
  std::map<ComponentId,int> combined_depths;

  for(const FaceId final_fid : graph.faces()) {
    if(face_info[final_fid] != FaceEnclosedBy::Unknown) continue;
    path_to_infinity.clear();
    FaceId fid = final_fid;
    // Traverse from target face back to boundary recording crossings
    while(fid != boundary_face) {
      const auto crossed_edge = crossing_info[fid].next;
      path_to_infinity.append(crossed_edge);
      assert(graph.face(crossed_edge) == fid);
      assert(graph.opp_face(crossed_edge) != fid);
      fid = graph.opp_face(crossed_edge);
    }
    combined_depths.clear();
    // Traverse backwards along path to target face saving crossing info as we go
    while(!path_to_infinity.empty()) {
      const auto fwd_edge = path_to_infinity.pop();
      const auto rev_edge = graph.reverse(fwd_edge);
      for(const ComponentId src : edge_components[fwd_edge]) ++combined_depths[src];
      for(const ComponentId src : edge_components[rev_edge]) {
        auto iter = combined_depths.find(src);
        assert(iter != combined_depths.end());
        --iter->second;
        if(iter->second == 0) combined_depths.erase(iter);
      }
      const auto enclosing_components = combined_depths.size();
      assert(std::all_of(combined_depths.cbegin(), combined_depths.cend(), [](const std::map<ComponentId,int>::value_type& v) {
        return (v.second >= 1);
       }));
      const FaceId face = graph.face(fwd_edge);
      if(enclosing_components == 0) {
        assert(face_info[face] == FaceEnclosedBy::Unknown || face_info[face] == FaceEnclosedBy::Zero);
        face_info[face] = FaceEnclosedBy::Zero;
      }
      else if(enclosing_components == 1) {
        assert(face_info[face] == FaceEnclosedBy::Unknown || face_info[face] == FaceEnclosedBy::One);
        face_info[face] = FaceEnclosedBy::One;
      }
      else {
        assert(enclosing_components >= 2);
        assert(face_info[face] == FaceEnclosedBy::Unknown || face_info[face] == FaceEnclosedBy::TwoOrMore);
        face_info[face] = FaceEnclosedBy::TwoOrMore;
      }
    }
  }

  auto overlap_faces = Field<bool, FaceId>{graph.n_faces()};
  for(const FaceId f : graph.faces()) {
    overlap_faces[f] = (face_info[f] == FaceEnclosedBy::TwoOrMore);
  }

  return overlap_faces;
}
#endif

Nested<CircleArc> find_overlapping_offsets(const Nested<const CircleArc> arcs, const real d) {
  const auto bounds = approximate_bounding_box(arcs).thickened(d);
  const auto quant = make_arc_quantizer(bounds);
  const Quantized signed_offset = quantize_offset(quant, d);
  if(signed_offset == 0) return Nested<CircleArc>{};
  const Quantized abs_offset = abs(signed_offset);
  // Build exact representation if inputs
  IntervalScope scope;
  VertexSet<PS> input_verts;
  auto input_arcs = input_verts.quantize_circle_arcs(quant, arcs);
  const auto input_graph = new_<PlanarArcGraph<PS>>(input_verts, input_arcs);

  // Find interior faces
  Field<bool, FaceId> input_interior_faces = faces_greater_than(*input_graph, 0);
  if(signed_offset < 0) {
    // If offset is negative, swap interior/exterior
    for(auto& is_interior : input_interior_faces.flat) is_interior = !is_interior;
  }

  // Group faces into components
  const Field<ComponentId, FaceId> input_face_components = cluster_faces(input_graph->topology, input_interior_faces);

  // Pull out new borders as (half-)edges
  const auto input_contour_edges = extract_region(input_graph->topology, input_interior_faces);

  // Convert to contours and simplify any concentric arcs
  const auto input_contours = input_graph->combine_concentric_arcs(input_graph->edges_to_closed_contours(input_contour_edges));
  // input_contours should correspond to input_contour_edges, just with a chance of fewer primitives
  // TODO: Review if combine_concentric_arcs is worth calling here. I suspect combine_concentric_arcs is a waste of effort
  //   here since union isn't likely to produce concentric arcs

  // Map contours back to components
  auto input_contour_components = Array<ComponentId>{input_contour_edges.size(), uninit};
  for(const int i : range(input_contour_edges.size())) {
    // Every halfedge in the contour should belong to the same component so use the first one
    const HalfedgeId hid0 = input_contour_edges[i].front();
    input_contour_components[i] = input_face_components[input_graph->topology->face(hid0)];
  }

  // At this point, we have grouped input into components
  // We want to know where offsets of multiple components overlap with each other

  // We will build offset of the inputs as in a normal arc offset, but we also track which component each added contour came from
  ArcAccumulator<PS> minkowski_terms;
  Array<ComponentId> minkowski_term_sources;

  // Since we just extracted components via splitting above, we know that components don't overlap with each other initially
  // This means we just need to check for overlap between their borders
  // As a result, we don't need to include the original input geometry in our offset (though this isn't that much easier)

  for(const int i : range(input_contours.size())) {
    const auto num_previous_contours = minkowski_terms.contours.size();
    for(const auto sa : input_contours[i]) {
      const auto ccw_a = input_graph->vertices.arc(input_graph->vertices.ccw_arc(sa));
      // Add a capsule around the arc
      add_capsule(minkowski_terms, ccw_a, abs_offset); // Use abs_offset since we swapped inside/outside above
    }
    // Mark source for each new contour added as part of a capsule
    const auto num_new_contours = minkowski_terms.contours.size() - num_previous_contours;
    minkowski_term_sources.extend(ConstantMap<ComponentId>{num_new_contours, input_contour_components[i]});
  }

  const auto offset_graph = minkowski_terms.compute_embedding();
  const NestedField<ComponentId,HalfedgeId> offset_edge_sources = flag_contour_halfedges<ComponentId>(offset_graph, minkowski_terms.contours, minkowski_term_sources);

  const auto overlap_faces = find_overlapping_components(offset_graph->topology, offset_edge_sources, offset_graph->boundary_face());
  assert(overlap_faces.flat == find_overlapping_components_slow(offset_graph->topology, offset_edge_sources, offset_graph->boundary_face()).flat);

  return offset_graph->unquantize_circle_arcs(quant,extract_region(offset_graph->topology, overlap_faces));
}

} // namespace geode