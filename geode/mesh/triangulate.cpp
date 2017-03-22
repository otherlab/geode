#include "triangulate.h"
#include <geode/array/amap.h>
#include <geode/exact/delaunay.h>
#include <geode/exact/polygon_csg.h>
#include <geode/exact/quantize.h>

namespace geode {

static Tuple<Ref<TriangleTopology>,Field<Vec2,VertexId>> generate_debug_mesh(const Quantizer<real,2>& quant, 
                                                                             const RawArray<const Vector<Quantized,2>> x,
                                                                             const DelaunayConstraintConflict& e) {
  Field<Vec2,VertexId> debug_x;
  Ref<MutableTriangleTopology> topology = new_<MutableTriangleTopology>();
  topology->add_vertices(8);
  topology->add_face(Vector<VertexId,3>{vec(0,1,2)});
  topology->add_face(Vector<VertexId,3>{vec(1,0,3)});
  topology->add_face(Vector<VertexId,3>{vec(4,5,6)});
  topology->add_face(Vector<VertexId,3>{vec(5,4,7)});
  const auto e00 = quant.inverse(x[e.e0.x]);
  const auto e01 = quant.inverse(x[e.e0.y]);
  const auto e10 = quant.inverse(x[e.e1.x]);
  const auto e11 = quant.inverse(x[e.e1.y]);
  debug_x.append(e00);
  debug_x.append(e01);
  debug_x.append((e00 + e01)*0.5 + rotate_left_90(e01 - e00)*0.5);
  debug_x.append((e00 + e01)*0.5 - rotate_left_90(e01 - e00)*0.5);
  debug_x.append(e10);
  debug_x.append(e11);
  debug_x.append((e10 + e11)*0.5 + rotate_left_90(e11 - e10)*0.5);
  debug_x.append((e10 + e11)*0.5 - rotate_left_90(e11 - e10)*0.5);
  return {topology, debug_x}; 
}

static Array<Vector<int,2>> make_edges(const Nested<Vector<Quantized,2>>& polygons) {
  const auto edges = Array<Vector<int,2>>{polygons.total_size(), uninit};
  for(const int i : range(polygons.total_size())) {
    edges[i][0] = i;
    edges[i][1] = i+1;
  }
  for(const int n : range(polygons.size())) {
    const int i_front = polygons.offsets[n];
    const int i_back = polygons.offsets[n+1] - 1;
    edges[i_back][1] = i_front;
  }
  return edges;
}

// Clean up faces around non-convex polygons or between disconnected polygons
static void erase_faces_outside_polygon(MutableTriangleTopology& topology, const RawArray<const Vector<int,2>> input_edges) {

  // This function assumes edges form a bunch of simple polygon. See is_input_boundary for more assumptions
  assert(topology.n_vertices() == topology.allocated_vertices() && topology.n_vertices() == input_edges.size());

  const auto is_input_boundary = [&](const HalfedgeId hid) {
    const int src = topology.src(hid).idx();
    const int dst = topology.dst(hid).idx();
    // This function assumes edges are from polygon and arranged such that we can map each vertex to it's unique outgoing edge
    // For more general sets of edges we would probably need to add a field to topology or something
    assert(input_edges[src].x == src);
    assert(input_edges[dst].x == dst);
    assert(input_edges[src].y == (input_edges[src].x + 1) || input_edges[src].y < input_edges[src].x);
    assert(input_edges[dst].y == (input_edges[dst].x + 1) || input_edges[dst].y < input_edges[dst].x);
    return (input_edges[src].y == dst) || (input_edges[dst].y == src);
  };

  // Find inside/outside by looking at input_edges and floodfill to erase all triangles that aren't inside input_edges

  Field<bool, FaceId> to_be_erased{topology.allocated_faces()};
  Array<FaceId> cleanup_queue;

  // Initialize queue with any faces that are outside of input edges
  for(const HalfedgeId hid : topology.interior_halfedges()) {
    const int src = topology.src(hid).idx();
    const int dst = topology.dst(hid).idx();
    const auto fid = topology.face(hid);
    // Find backwards edges in input and mark those as needing to be erased
    if(!to_be_erased[fid] && input_edges[dst].y == src) {
      cleanup_queue.append(fid);
      to_be_erased[fid] = true;
    }
  }

  // Floodfill up to input edges
  while(!cleanup_queue.empty()) {
    for(const HalfedgeId hid : topology.halfedges(cleanup_queue.pop())) {
      if(is_input_boundary(hid)) continue; // Don't propagate across any input edges
      assert(!topology.is_boundary(hid)); // Boundary edges shouldn't belong to a face
      const auto opp = topology.reverse(hid);
      if(topology.is_boundary(opp)) continue;
      const auto opp_face = topology.face(opp);
      if(!to_be_erased[opp_face]) {
        to_be_erased[opp_face] = true;
        cleanup_queue.append(opp_face);
      }
    }
  }

  for(const FaceId fid : to_be_erased.id_range()) {
    if(to_be_erased[fid]) {
      topology.erase(fid);
    }
  }
  assert(topology.n_vertices() == topology.allocated_vertices());
  topology.collect_garbage();
}

Tuple<Ref<TriangleTopology>,Field<Vec2,VertexId>> triangulate_polygon(const Nested<Vec2>& raw_polygons) {

  const auto quant = quantizer(bounding_box(raw_polygons));


  auto exact_union = exact_split_polygons_with_rule(amap(quant, raw_polygons), 0, FillRule::Greater);
  auto edges = make_edges(exact_union);
  
  constexpr int max_attempts = 10;

  for(int attempt = 0;; ++attempt) {
    try {
      Ref<MutableTriangleTopology> topology = exact_delaunay_points(exact_union.flat, edges)->mutate();
      // exact_delaunay_points triangulates the full convex hull. This functions cleans up anything that wasn't inside the actual polygons
      erase_faces_outside_polygon(topology, edges);
      return {topology, Field<Vec2,VertexId>{amap(quant.inverse,exact_union.flat).copy()}};
    } catch(const DelaunayConstraintConflict& e) {
      if(attempt < max_attempts) {
        // exact_split_polygons_with_rule should resolve most self intersections, but could create new ones when it approximates intersections
        // I've been working on an algorithm that simplifies newly created self intersections, but it's not finished yet
        // For now we try splitting multiple times which should resolve vaguely reasonable cases
        // Short of implementing provably self intersection free approximations, just merging nearly coincident points would probably make this work
        exact_union = exact_split_polygons_with_rule(exact_union, 0, FillRule::Greater);
        edges = make_edges(exact_union);
        continue;
      }
      else {
        // Return a mesh that shows where the problem is
        return generate_debug_mesh(quant, exact_union.flat, e);
      }
    }
  }
}

} // geode namespace