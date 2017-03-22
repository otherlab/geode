#include "refine_mesh.h"
#include <geode/geometry/polygon.h>
#include <geode/random/permute.h>

// It should be safe to remove calls to check_interrupts once this is tested some more
#include <geode/utility/interrupts.h>
namespace other {

// Brute force version of split_long_edges that always splits longest edge in the entire mesh
// It's easier to reason about why this should work and it produces slightly better looking meshes, but as implemented it is O(N^2)
// I'm leaving this here for now to be used as a reference if we find problem cases for the faster version
GEODE_UNUSED static void split_long_edges_slow_but_safe(MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x, const real max_edge_length2) {
  for(HalfedgeId longest_splittable_edge;;) {
    longest_splittable_edge = HalfedgeId{};
    real longest_edge_length2 = max_edge_length2;
    for(const HalfedgeId hid : mesh.halfedges()) {
      check_interrupts();
      const auto rev_hid = mesh.reverse(hid);
      if(mesh.is_boundary(rev_hid))
        continue;
      if(rev_hid < hid)
        continue;
      const auto segment = mesh.segment(x, hid);
      const auto segment_length2 = segment.sqr_length();
      if(!(segment_length2 > longest_edge_length2))
        continue; // Segment is already short enough. Skip it
      longest_edge_length2 = segment_length2;
      longest_splittable_edge = hid;
    }
    if(!longest_splittable_edge.valid())
      break;
    const auto segment = mesh.segment(x, longest_splittable_edge);
    const auto new_vid = mesh.split_edge(longest_splittable_edge);
    x[new_vid] = segment.center();
  }
}

static bool is_convex_quad(const Vector<Vec2,4> x) {
  const Vector<Vec2,4> edges = x - x.roll(1);
  for(const int curr_i : range(4)) {
    const int next_i = (curr_i+1) % 4;
    if(cross(edges[curr_i], edges[next_i]) <= 0) {
      return false;
    }
  }
  return true;
}

static real longest_distance2(const Vector<Vec2,4> x, const Vec2 p) {
  real max_dist2 = (x[0] - p).sqr_magnitude();
  for(const int i : range(1,4)) {
    real new_dist2 = (x[i] - p).sqr_magnitude();
    if(new_dist2 > max_dist2) max_dist2 = new_dist2;
  }
  return max_dist2;
}

struct EdgeAndLength {
  EdgeAndLength() = default;
  HalfedgeId hid;
  real sqr_length;
  inline friend bool operator<(const EdgeAndLength& lhs, const EdgeAndLength& rhs) {
    return rhs.sqr_length < lhs.sqr_length; // Sort longest to shortest
  }
};

// Checks edges of a face and splits them if sqr_length > max_edge_sqr_length
// Returns true if any edge was split
// If allow_adjacent is true, might split edge on a neighboring face if needed to make sure splitting shared edge doesn't lead to cycle of creating edges that need splitting
// Never splits more than one edge. This function should be called on the same face until it returns false
static bool try_split_face(const FaceId fid, const bool allow_adjacent, MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x, const real max_edge_sqr_length) {

  const auto edge_sqr_length = [&](const HalfedgeId hid)
  { return mesh.segment(x,hid).sqr_length(); };

  const auto get_edge = [&](const HalfedgeId hid)
  { return EdgeAndLength{hid, edge_sqr_length(hid)}; };

  // Get edges in decreasing length
  const auto halfedges = mesh.halfedges(fid);
  const auto ordered_edges = vec(get_edge(halfedges[0]),get_edge(halfedges[1]),get_edge(halfedges[2])).sorted();

  for(const auto edge_and_length : ordered_edges) {
    const auto curr_sqr_length = edge_and_length.sqr_length;
    if(!(curr_sqr_length > max_edge_sqr_length)) {
      return false; // Edge is already short enough so we don't need to do anything and remaining edges will be even shorter
    }
    const HalfedgeId hid = edge_and_length.hid;
    assert(mesh.valid(hid)); // All of these should remain valid
    assert(!mesh.is_boundary(hid)); // We should only iterate over interior edges

    const auto rev_hid = mesh.reverse(hid);
    if(mesh.is_boundary(rev_hid)) {
      // It should always be safe to split boundary edges
      const auto midpoint = mesh.segment(x, hid).center();
      const auto new_vid = mesh.split_edge(hid);
      x[new_vid] = midpoint;
      return true;
    }

    // Only look at each edge from the face with the lower id
    // Compare halfedge ids directly instead of extracting face ids
    if(allow_adjacent && rev_hid < hid) {
      // Since we process faces in id order, we should already have split this edge when looking at opposite face
      // I think the only way we should get here is if edge is very close to splitting threshold and we rounded differently when checking this side
      assert(curr_sqr_length <= max_edge_sqr_length + 1e-6);
      continue;
    }
    assert(!allow_adjacent || mesh.face(hid) < mesh.face(rev_hid));

    // We could split at midpoint of segment, but newly created edges could be nearly as long or longer than the current edge
    // Iterating in a bad order can lead to a long cycle of splitting newly created edges that are each only slightly shorter than before
    // Since we start with longest edges on each face we can only be creating longer edges on the opposite face
    if(allow_adjacent) {
      assert(mesh.face(rev_hid).valid()); // If we are on boundary, should have caught that above
      const auto opp_n_sqr_length = edge_sqr_length(mesh.next(rev_hid));
      const auto opp_p_sqr_length = edge_sqr_length(mesh.prev(rev_hid));
      if(curr_sqr_length < opp_n_sqr_length || curr_sqr_length < opp_p_sqr_length) {
        // This isn't the longest edge on the opposite face
        // Try to split one of those edges first which should ensure we aren't creating edges faster than we're splitting them
        // Note: I haven't worked out a proof that this is sufficient, but I haven't found a counter example
        if(try_split_face(mesh.face(rev_hid), false, mesh, x, max_edge_sqr_length)) {
          return true;
        }
      }
    }
    
    const auto segment = mesh.segment(x, hid);
    const auto new_vid = mesh.split_edge(hid);
    assert(x.valid(new_vid));
    x[new_vid] = segment.center();
    return true;
  }
  return false;
}

static void split_long_edges(MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x, const real max_edge_sqr_length) {
  // Splitting modifies two existing faces and creates two new ones
  // The new faces will have larger ids than any existing face and will therefore be checked later just by iterating over ids
  // The other modified face was mesh.face(rev_hid), but we can't have checked that face yet or we would have split that edge already
  // Thus we can assume it comes after the current face and will be checked later
  // (try_split_face also specifically skips edges by looking at opposite ids)
  // Warning: We need to catch new faces added to mesh after this loop starts so we call all_faces().end() on every iteration in loop instead of a normal range based for loop
  for(auto fid_iter = mesh.all_faces().begin(); fid_iter != mesh.all_faces().end(); ++fid_iter) {
    const auto fid = *fid_iter;
    while(mesh.valid(fid) && try_split_face(fid, true, mesh, x, max_edge_sqr_length)) {
      check_interrupts(); // Until this is more thoroughly tested, make it easy to kill app if we're stuck in an infinite splitting cycle
      continue;
    }
  }
}

static void collapse_short_edges(MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x, const real min_edge_length2, const real max_edge_length2) {
  // We maintain list of dirty faces to be checked. This causes a bunch of redundant checks, but drastically simplifies bookkeeping
  Array<FaceId> queue; // Queue should never contain redundant copies of the same face
  Field<bool, FaceId> pending = Field<bool, FaceId>{mesh.allocated_faces()};

  // Start by adding all faces to the queue
  for(const FaceId fid : mesh.faces()) {
    queue.append(fid);
    assert(pending.valid(fid));
    pending[fid] = true;
  }

  const auto collapse_ok = [&](const Vector<VertexId,2> verts, const Vec2 new_candidate_point) {
    for(const int i : range(2)) {
      const VertexId v0 = verts[i];
      const VertexId opp_vid = verts[1-i];
      const auto x0 = x[v0];
      for(const HalfedgeId hid : mesh.outgoing(v0)) {
        const VertexId v1 = mesh.dst(hid);
        if(v1 == opp_vid) continue;
        if(!((x0 - x[v1]).sqr_magnitude() < max_edge_length2)) {
          return false;
        }
        // Check if new triangle would now have negative area
        const auto x1 = x[v1];
        const auto x2 = x[mesh.dst(mesh.next(hid))];
        if(cross(x1 - x0, x2 - x0) < 0.) {
          return false;
        }
      }
    }
    return true;
  };

  while(!queue.empty()) {
    FaceId next_fid = queue.pop();
    if(!mesh.valid(next_fid)) continue;
    assert(pending.valid(next_fid));
    assert(pending[next_fid]);
    pending[next_fid] = false;

    for(const HalfedgeId hid : mesh.halfedges(next_fid)) {
      if(mesh.is_boundary(mesh.reverse(hid))) continue; // Ignore edges along boundary
      if(!mesh.is_collapse_safe(hid)) continue; // Ignore edges that aren't topologically safe to collapse
      const auto src = mesh.src(hid);
      const auto dst = mesh.dst(hid);
      if(mesh.is_boundary(src) && mesh.is_boundary(dst))
        continue; // Don't collapse edges that cut between different boundaries
      const auto segment = mesh.segment(x, hid);
      if(!(segment.sqr_length() < min_edge_length2))
        continue; // Segment is already long enough. Skip it
      const auto verts = mesh.vertices(hid);
      auto new_candidate_point = segment.center(); // TODO: Be smarter about this choice to avoid flipping faces
      // If one end of edge is on boundary, don't move that end
      if(mesh.is_boundary(src)) {
        new_candidate_point = segment.x0;
      } 
      else if(mesh.is_boundary(dst)) {
        new_candidate_point = segment.x1;
      }
      if(!collapse_ok(verts, new_candidate_point))
        continue;
      const auto collapsed_faces = mesh.faces(hid);
      // Grab all faces that are about to be changed and add them
      for(const VertexId vid : verts) {
        for(const HalfedgeId hid : mesh.outgoing(vid)) {
          const FaceId fid = mesh.face(hid);
          if(!mesh.valid(fid))
            continue;
          assert(pending.valid(fid));
          if(collapsed_faces.contains(fid) || pending[fid])
            continue;
          pending[fid] = true;
          queue.append(fid);
          assert(queue.size() <= mesh.allocated_faces()); // Minimal check that we aren't adding faces more than once
        }
      }
      x[verts[0]] = new_candidate_point;
      x[verts[1]] = new_candidate_point;
      assert(!mesh.is_boundary(hid) && !mesh.is_boundary(mesh.reverse(hid)));
      mesh.unsafe_collapse(hid);
      break; // Skip remaining halfedges on this face since it just got erased
    }
  }
}

static void equalize_valences(MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x) {
  // TODO: Iterate over each edge once. This currently doesn't since flip_edge shuffles around ids
  for(const HalfedgeId hid : mesh.interior_halfedges()) {
    if(!mesh.is_flip_safe(hid))
      continue;
    assert(!mesh.is_boundary(mesh.reverse(hid))); // In this case is_flip_safe should be false and we should have skipped this edge
    const Vector<VertexId,4> quad = vec(mesh.dst(hid),mesh.opposite(hid),mesh.src(hid),mesh.opposite(mesh.reverse(hid)));
    if(!is_convex_quad(x.vec(quad)))
      continue; // Avoid flipping non-convex quads since flipped edge will extend outside boundary of mesh
    Vector<int,4> valence;
    Vector<int,4> opt_valence;
    for(const int i : range(4)) {
      valence[i] = mesh.valence(quad[i]);
      opt_valence[i] = mesh.is_boundary(quad[i]) ? 4 : 6;
    }
    const Vector<int,4> flipped_valence = valence + vec(-1,1,-1,1);
    if((flipped_valence - opt_valence).sqr_magnitude() < (valence - opt_valence).sqr_magnitude()) {
      GEODE_UNUSED auto unused = mesh.flip_edge(hid);
    }
  }
}

static const uint128_t key = uint128_t(9975794406056834021u)+(uint128_t(920519151720167868u)<<64);
// This key is reused from delaunay.cpp and mesh_csg.cpp

static void smooth_internal_vertices(const MutableTriangleTopology& mesh, const Field<Vec2,VertexId>& x, const int seed) {
  const int n = mesh.allocated_vertices();
  const uint128_t permutation = key - seed;
  for(const int i : range(n)) {
    // Iterate over vertices in a random order to minimize side effects of sweeping in a specific order
    const VertexId vid = VertexId{static_cast<int>(random_permute(n,permutation,i))};
    if(!mesh.valid(vid) || mesh.is_boundary(vid)) continue;
    Vec2 sum;
    int count = 0;
    for(const HalfedgeId hid : mesh.outgoing(vid)) {
      sum += x[mesh.dst(hid)];
      ++count;
    }
    if(count != 0) {
      x[vid] = sum/count;
    }
  }
}

void refine_mesh(MutableTriangleTopology& mesh, const FieldId<Vec2,VertexId> x_id, const real target_edge_length, const int iterations) {

  const real min_edge_length = (4./5.)*target_edge_length;
  const real max_edge_length = (4./3.)*target_edge_length;
  const real min_edge_length2 = sqr(min_edge_length);
  const real max_edge_length2 = sqr(max_edge_length);

  const Field<Vec2,VertexId>& x = mesh.field(x_id);
  for(const int i : range(iterations)) {
    // Split edges longer the max_edge_length
    split_long_edges(mesh, x, max_edge_length2);
    // Collapse edges shorter than max_edge_length
    collapse_short_edges(mesh, x, min_edge_length2, max_edge_length2);
    // Flip edges to get closer to valence 6
    equalize_valences(mesh, x);
    // Smooth out uneven features in interior
    smooth_internal_vertices(mesh, x, i);
  }

  mesh.collect_garbage();
}

} // other namespace