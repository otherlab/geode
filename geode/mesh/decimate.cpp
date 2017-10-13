// Quadric-based mesh decimation

#include <geode/mesh/decimate.h>
#include <geode/geometry/ParticleTree.h>
#include <geode/mesh/quadric.h>
#include <geode/python/wrap.h>
#include <geode/structure/Heap.h>

namespace geode {

typedef real T;
typedef Vector<T,3> TV;

namespace {

enum class CollapseRank {
  simple,
  degenerate_faces,
  requires_split,
  unset, // Placeholder value less than not_allowed but greater than all others
  not_allowed
};

// For debugging
GEODE_UNUSED static std::ostream& operator<<(std::ostream& os, const CollapseRank rank) {
  switch(rank) {
    case CollapseRank::simple: return os << "simple"; 
    case CollapseRank::degenerate_faces: return os << "degenerate_faces"; 
    case CollapseRank::requires_split: return os << "requires_split"; 
    case CollapseRank::unset: return os << "unset"; 
    case CollapseRank::not_allowed: return os << "not_allowed"; 
  }
  return os << "<invalid-enum>";
}

struct CollapsePriority {
  CollapseRank rank; // Sort by rank
  T cost; // Then by value
  inline friend bool operator<(const CollapsePriority& lhs, const CollapsePriority& rhs) {
    return (lhs.rank == rhs.rank) ? lhs.cost < rhs.cost
                                  : lhs.rank < rhs.rank;
  }
  inline friend bool operator==(const CollapsePriority& lhs, const CollapsePriority& rhs) {
    return (lhs.rank == rhs.rank) && (lhs.cost == rhs.cost);
  }
  inline friend std::ostream& operator<<(std::ostream& os, const CollapsePriority& cp) {
    return os << "{" << cp.rank << ", " << cp.cost << "}";
  }
};

// Binary heap of potential collapses
struct VertexHeap : public HeapBase<VertexHeap>, public Noncopyable {
  typedef HeapBase<VertexHeap> Base;
  Array<Tuple<VertexId,CollapsePriority,VertexId>> heap; // src,badness,dst
  Field<int,VertexId> inv_heap;

  VertexHeap(const int nv)
    : inv_heap(nv,uninit) {
    inv_heap.flat.fill(-1);
  }

  int size() const {
    return heap.size();
  }

  bool first(const int i, const int j) const {
    // Sort by quality, but catch ties to ensure we don't get duplicates
    return heap[i].y < heap[j].y || ((heap[i].y == heap[j].y) && (heap[i].x < heap[j].x));
  }

  void swap(const int i, const int j) {
    std::swap(heap[i],heap[j]);
    inv_heap[heap[i].x] = i;
    inv_heap[heap[j].x] = j;
  }

  Vector<VertexId,2> pop() {
    const auto e = heap[0];
    inv_heap[e.x] = -1;
    const auto p = heap.pop();
    if (size()) {
      heap[0] = p;
      inv_heap[heap[0].x] = 0;
      Base::move_downward(0);
    }
    return vec(e.x,e.z);
  }

  void set(const VertexId v, const CollapsePriority q, const VertexId dst) {
    const auto entry = tuple(v,q,dst);
    // Grow heap as needed if id is out of range
    if(v.id >= inv_heap.size()) {
      const int old_size = inv_heap.size();
      const int new_size = v.idx() + 1;
      inv_heap.flat.resize(new_size);
      inv_heap.flat.slice(old_size, new_size).fill(-1);
    }
    int& i = inv_heap[v];
    if (i < 0)
      i = heap.append(entry);
    else
      heap[i] = entry;
    Base::move_up_or_down(i);
  }

  void erase(const VertexId v) {
    if(v.id >= inv_heap.size())
      return; // Value was never set so we can ignore it
    int& i = inv_heap[v];
    if (i >= 0) {
      const auto p = heap.pop();
      if (i < size()) {
        heap[i] = p;
        inv_heap[p.x] = i;
        Base::move_up_or_down(i);
      }
      i = -1;
    }
  }
};
} // anonymous namespace

// Check if the mesh is managing a field to ensure it will be updated when allocating new vertices
static GEODE_UNUSED bool mesh_owns_field(const MutableTriangleTopology& mesh, const Field<TV, VertexId>& X)
{ return mesh.find_field(X).valid(); } // Check for existing ID
static GEODE_UNUSED bool mesh_owns_field(const MutableTriangleTopology& mesh, const RawField<const TV, VertexId>& X)
{ return false; } // RawField becomes invalidated if underlying buffer gets reallocated

static Vector<HalfedgeId,3> find_loop(const MutableTriangleTopology& mesh, const HalfedgeId e01) {
  const HalfedgeId e10 = mesh.reverse(e01);
  // Look up left and right vertices
  const auto vl = mesh.is_boundary(e01) ? VertexId{} : mesh.dst(mesh.next(e01));
  const auto vr = mesh.is_boundary(e10) ? VertexId{} : mesh.dst(mesh.next(e10));

  const VertexId v0 = mesh.src(e01);
  const VertexId v1 = mesh.dst(e01);

  Hashtable<VertexId, HalfedgeId> v0_neighbors;
  for (const HalfedgeId e : mesh.outgoing(v0)) {
    const VertexId v = mesh.dst(e);
    if(v != vl && v != vr) {
      v0_neighbors.set(v, e);
    }
  }

  for(const HalfedgeId e : mesh.outgoing(v1)) {
    const VertexId v = mesh.dst(e);
    const HalfedgeId e02 = v0_neighbors.get_default(v);
    if(e02.valid()) {
      return {e01, e, mesh.reverse(e02)}; // e01, e12, e20
    }
  }

  return {};
}

enum class ReduceMode {
  decimate_only,
  simplify_topology
};
static constexpr bool splitting_enabled(const ReduceMode reduce_mode)
{ return reduce_mode == ReduceMode::simplify_topology; }

static CollapseRank collapse_rank(const MutableTriangleTopology& mesh, const HalfedgeId e01, const bool is_boundary) {
  if(mesh.is_collapse_safe(e01)) return CollapseRank::simple;
  if(!is_boundary) {
    const HalfedgeId e10 = mesh.reverse(e01);
    const VertexId vl = mesh.opposite(e01);
    const VertexId vr = mesh.opposite(e10);
    if(vl == vr) {
      return CollapseRank::degenerate_faces;
    }
    return CollapseRank::requires_split;
  }
  return CollapseRank::not_allowed;
}

template<ReduceMode reduce_mode, class TField> static void mesh_reduce_helper(MutableTriangleTopology& mesh, const TField& X,
                      const T distance, const T max_angle, const int min_vertices, const T boundary_distance) {
  if (mesh.n_vertices() <= min_vertices)
    return;

  if(splitting_enabled(reduce_mode)) {
    mesh.erase_isolated_vertices();
  }
  // If splitting is enabled, make sure X is managed by the mesh and will be updated accordingly
  assert(!splitting_enabled(reduce_mode) || mesh_owns_field(mesh, X));

  const T area = sqr(distance);
  // We consider face to be degenerate if face area is smaller than sqr(distance) (this is a mostly arbitrary choice, but at least the units match)
  // area_of_face = 0.5*magnitude(n) < sqr(distance)
  // sqr_magnitude(n) < 4.*sqr(sqr(distance))
  const T sqr_magnitude_n_eps = 4.*sqr(area); // Threshold for testing if faces are degenerate
  const T sign_sqr_min_cos = sign_sqr(max_angle > .99*pi ? -1 : cos(max_angle));

  const auto collapse_changes_normal_too_much = [&mesh, &X, sign_sqr_min_cos, sqr_magnitude_n_eps](const HalfedgeId e) {
    if (sign_sqr_min_cos > -1) {
      const VertexId vs = mesh.src(e);
      const VertexId vd = mesh.dst(e);
      const auto xs = X[vs],
                 xd = X[vd];

      // Cached computation of src vertex normal since we are likely to need it several times or not at all
      bool src_vertex_normal_cached = false;
      TV cached_src_vertex_normal;
      const auto src_vertex_normal = [&]() {
        if(!src_vertex_normal_cached) {
          cached_src_vertex_normal = mesh.normal(X,vs);
          src_vertex_normal_cached = true;
        }
        return cached_src_vertex_normal;
      };

      for (const auto ee : mesh.outgoing(vs)) {
        if (e!=ee && !mesh.is_boundary(ee)) {
          const auto v2 = mesh.opposite(ee);
          if (v2 != vd) {
            const auto x1 = X[mesh.dst(ee)],
                       x2 = X[v2];
            auto n0 = cross(x2-x1,xs-x1);
            const auto n1 = cross(x2-x1,xd-x1);
            auto sqr_magnitude_n0 = sqr_magnitude(n0);
            const auto sqr_magnitude_n1 = sqr_magnitude(n1);
            // Does collapse turn a degenerate face into a non-degenerate face?
            if(sqr_magnitude_n0 <= sqr_magnitude_n_eps && !(sqr_magnitude_n1 <= sqr_magnitude_n_eps)) {
              // If so, we don't have an original normal, but we still need to make sure new face isn't 'flipped'
              // I'm not sure what the best way to do this is, but without some sort of check here, concave areas near degenerate vertices can get filled in by thin sheets
              // I'm using average normal at the src vertex since it's robust and easy
              // Checking that dihedral angles at edges of new face are all less than max_angle might be better
              // However this would require updating neighbors two edges away after each collapse
              n0 = src_vertex_normal();
              sqr_magnitude_n0 = 1;
            }
            if (sign_sqr(dot(n0,n1)) < sign_sqr_min_cos*sqr_magnitude_n0*sqr_magnitude_n1) {
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  // Finds the best edge to collapse v along.  Returns (q(e),dst(e)).
  // Best edge to collapse doesn't require splitting any vertices if possible and has smallest error cost for quadric
  const auto best_collapse = [&mesh,&X,&collapse_changes_normal_too_much,area,boundary_distance](const VertexId v) {
    if(mesh.isolated(v)) {
      return tuple(CollapsePriority{},VertexId{});
    }
    Quadric q = compute_quadric(mesh,X,v);
    // Find the best edge, including normal constraints
    T min_q = inf;
    // If splitting isn't enabled, initialize min_rank to simple so that we discard any collapses that would require a split
    CollapseRank min_rank = splitting_enabled(reduce_mode) ? CollapseRank::unset
                                                           : CollapseRank::simple; 
    HalfedgeId min_e;

    const bool is_boundary = mesh.is_boundary(v);
    Vector<Segment<TV>,2> boundary_edges;
    if(is_boundary) {
      const HalfedgeId b = mesh.halfedge(v);
      boundary_edges[0] = mesh.segment(X, b);
      boundary_edges[1] = mesh.segment(X, mesh.prev(b));
    }
    for (const auto e : mesh.outgoing(v)) {
      const TV xd = X[mesh.dst(e)];
      const T qx = q(xd);
      if(!(qx <= area)) continue; // Error too big, skip remaining checks
      if(min_q <= qx && min_rank == CollapseRank::simple) continue; // Already found best possible rank and better quality so we can bypass some of the more expensive checks
      if(collapse_changes_normal_too_much(e)) continue; // Violates normal constraint
      // Are we moving a boundary vertex too far from its two boundary lines?
      if(is_boundary && (   line_point_distance(boundary_edges[0],xd) > boundary_distance
                         || line_point_distance(boundary_edges[1],xd) > boundary_distance))
        continue;
      const auto new_rank = collapse_rank(mesh, e, is_boundary);
      if(new_rank < min_rank || (new_rank == min_rank && qx < min_q)) {
        min_q = qx;
        min_rank = new_rank;
        min_e = e;
      }
    }
    return tuple(CollapsePriority{min_rank, min_q},
                 mesh.valid(min_e) ? mesh.dst(min_e) : VertexId{}); // Catch isolated vertices and ones we can't collapse
  };

  // TODO: Best collapse from a given vertex depends on every neighbor of every neighbor
  //   It might be faster to maintain a heap of halfedges only tracking error from quadrics so that normals and boundary distances don't get evaluated as often
  //   Need to be careful that an invalid collapse with a lower error doesn't hide another valid collapse

  // Initialize quadrics and heap
  VertexHeap heap(mesh.allocated_vertices());
  for (const auto v : mesh.vertices()) {
    const auto qe = best_collapse(v);
    if (qe.y.valid())
      heap.inv_heap[v] = heap.heap.append(tuple(v,qe.x,qe.y));
  }
  heap.make();

  // Update the quadric information for a vertex
  const auto update = [&heap,best_collapse,area](const VertexId v) {
    const auto qe = best_collapse(v);
    if (qe.y.valid())
      heap.set(v,qe.x,qe.y);
    else
      heap.erase(v);
  };

  Array<VertexId> dirty;

  // Repeatedly collapse the best vertex
  while (heap.size()) {
    const CollapseRank rank = heap.heap[0].y.rank;
    const auto v = heap.pop();
    const auto vs = v.x;
    const auto vd = v.y;
    assert(mesh.valid(vs) && mesh.valid(vd)); // Heap should be kept up to date as we erase vertices
    const auto e = mesh.halfedge(vs, vd);
    assert(e.valid()); // Halfedge should still exist
    assert(collapse_rank(mesh,e,mesh.is_boundary(vs)) == rank); // This shouldn't have changed since entry was added to heap
    assert(!collapse_changes_normal_too_much(e)); // Should have checked this when choosing next vertex
    assert(rank == CollapseRank::simple || splitting_enabled(reduce_mode)); // Make sure splitting is allowed
    if(rank == CollapseRank::simple) {
      assert(mesh.is_collapse_safe(e));
      mesh.unsafe_collapse(e);
      if (mesh.n_vertices() <= min_vertices)
        break;
      // Don't need to update vs since it should have just been popped from heap
      update(vd);
      for(const HalfedgeId e : mesh.outgoing(vd)) {
        update(mesh.dst(e));
      }
    }
    else if(rank == CollapseRank::degenerate_faces) {
      // Collapse turned a tetrahedron into a pair of degenerate faces
      assert(mesh.opposite(e) == mesh.opposite(mesh.reverse(e)));
      const VertexId vo = mesh.opposite(e);
      mesh.collapse_degenerate_face_pair(e);
      if (mesh.n_vertices() <= min_vertices)
        break;
      for(const VertexId v : {vo, vs, vd}) {
        if(mesh.erased(v)) {
          heap.erase(v);
        }
        else {
          for(const HalfedgeId e : mesh.outgoing(v)) {
            dirty.append_unique(mesh.dst(e));
          }
        }
      }
    }
    else {
      assert(rank == CollapseRank::requires_split);
      // There should be an intersection in one-rings of vs and vd that are preventing the collapse
      // We can fix this by splitting the loop of edges
      // If there are multiple intersections, we might have to repeat this multiple times before we can collapse
      const auto loop = find_loop(mesh, e);
      assert(loop.x == e);
      const auto new_faces = mesh.split_loop(loop.x, loop.y, loop.z);
      for(const FaceId f : new_faces) {
        for(const VertexId v : mesh.vertices(f)) {
          dirty.append_unique(v);
          for(const HalfedgeId e : mesh.outgoing(v)) {
            dirty.append_unique(mesh.dst(e));
          }
        }
      }
    }
    for(const VertexId v : dirty) {
      update(v);
      for (const auto e : mesh.outgoing(v)) {
        update(mesh.dst(e));
      }
    }
    dirty.clear();
  }
  // With splitting enabled, isolated vertices are erased
  // No new isolated vertices should be created
  assert(mesh.n_vertices() <= min_vertices || !splitting_enabled(reduce_mode) || !mesh.has_isolated_vertices());
}

Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>
decimate(const TriangleTopology& mesh, RawField<const TV,VertexId> X,
         const T distance, const T max_angle, const int min_vertices, const T boundary_distance) {
  const auto rmesh = mesh.mutate();
  const auto rX = X.copy();
  decimate_inplace(rmesh,rX,distance,max_angle,min_vertices,boundary_distance);
  return Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>(rmesh,rX);
}

void decimate_inplace(MutableTriangleTopology& mesh,
                 RawField<const Vector<real,3>,VertexId> X,
                 const real distance,
                 const real max_angle,
                 const int min_vertices,
                 const real boundary_distance) {
  mesh_reduce_helper<ReduceMode::decimate_only>(mesh, X, distance, max_angle, min_vertices, boundary_distance);
}

void simplify_inplace(MutableTriangleTopology& mesh,
                 const FieldId<Vector<real,3>,VertexId> X_id,
                 const real distance,
                 const real max_angle,
                 const int min_vertices,
                 const real boundary_distance) {
  mesh_reduce_helper<ReduceMode::simplify_topology>(mesh, mesh.field(X_id), distance, max_angle, min_vertices, boundary_distance);
} 

void simplify_inplace_python(MutableTriangleTopology& mesh,
                 const PyFieldId& X_id,
                 const real distance,
                 const real max_angle,
                 const int min_vertices,
                 const real boundary_distance) {
  GEODE_ASSERT(X_id.prim == PyFieldId::Vertex);
  GEODE_ASSERT(X_id.type && (*X_id.type == typeid(Vector<real,3>)));
  mesh_reduce_helper<ReduceMode::simplify_topology>(mesh, mesh.field(FieldId<Vector<real,3>,VertexId>{X_id.id}), distance, max_angle, min_vertices, boundary_distance);
}

GEODE_CORE_EXPORT Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
simplify(const TriangleTopology& mesh,
         RawField<const Vector<real,3>,VertexId> X,
         const real distance,
         const real max_angle,
         const int min_vertices,
         const real boundary_distance) {
  Ref<MutableTriangleTopology> rmesh = mesh.mutate();
  const auto X_id = rmesh->add_field(X.copy());
  simplify_inplace(rmesh, X_id, distance, max_angle, min_vertices, boundary_distance);
  return {new_<TriangleTopology>(rmesh), // Make a non-mutable TriangleTopology from rmesh. This should share rather than copy non-mutable parts and let mutable parts be released after this function returns
          rmesh->field(X_id)};
}

static Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> tetrahedron_mesh() {
  typedef Vector<int,3> IV;
  static const IV tris[] = {IV{2,1,0},IV{0,1,3},IV{1,2,3},IV{2,0,3}};
  static const TV X[] = {TV{ sqrt(8./9.),           0, -1./3.},
                         TV{-sqrt(2./9.), sqrt(2./3.), -1./3.},
                         TV{-sqrt(2./9.),-sqrt(2./3.), -1./3.},
                         TV{           0,           0,      1}};
  return tuple(new_<TriangleSoup>(asarray(tris).copy()),asarray(X).copy());
}

using MeshAndX = Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>;

static int count_unconnected_clusters(const MeshAndX& mesh_and_x, const T epsilon) {
  const auto& mesh = *mesh_and_x.x;
  const auto& X = mesh_and_x.y;

  Array<TV> non_erased_verts;
  Array<VertexId> ids;
  for(const VertexId vid : mesh.vertices()) {
    non_erased_verts.append(X[vid]);
    ids.append(vid);
  }

  const int n = non_erased_verts.size();
  GEODE_ASSERT(n == mesh.n_vertices());
  Array<int> duplicates = new_<ParticleTree<TV>>(non_erased_verts, 4)->remove_duplicates(epsilon);
  GEODE_ASSERT(non_erased_verts.size() == n);
  GEODE_ASSERT(duplicates.size() == n);

  Hashtable<int, Array<VertexId>> clusters;
  for(const int i : range(n)) {
    clusters[duplicates[i]].append(ids[i]);
  }

  int unconnected_clusters = 0;
  for(const auto& index_and_cluster : clusters) {
    const Array<VertexId>& cluster = index_and_cluster.y;
    GEODE_ASSERT(!cluster.empty());
    Hashtable<VertexId> neighbors;
    for(const HalfedgeId e : mesh.outgoing(cluster.front())) neighbors.set(mesh.dst(e));
    neighbors.set(cluster.front());
    for(const VertexId v : cluster) {
      if(!neighbors.contains(v)) {
        unconnected_clusters += 1;
        break;
      }
    }
  }

  return unconnected_clusters;
}
static int count_degenerate_edges(const MeshAndX& mesh_and_x, const T epsilon) {
  const auto& mesh = *mesh_and_x.x;
  const auto& X = mesh_and_x.y;
  int degenerate_edges = 0;
  for(const HalfedgeId hid : mesh.halfedges()) {
    if(mesh.reverse(hid) > hid) continue; // Only visit each edge once
    if(mesh.edge_length(X, hid) < epsilon) {
      degenerate_edges += 1;
    }
  }
  return degenerate_edges;
}

static bool has_duplicate_faces(const MeshAndX& mesh_and_x) {
  const auto& mesh = *mesh_and_x.x;
  Hashtable<Vector<VertexId,3>,FaceId> face_map;
  for(const FaceId f : mesh.faces()) {
    const auto f_verts = mesh.vertices(f).sorted();
    if(face_map.contains(f_verts)) {
      return true;
    }
    face_map.set(f_verts, f);
  }
  return false;
}

static bool operator!=(const TriangleTopology::FaceInfo& lhs, const TriangleTopology::FaceInfo& rhs) {
  return (lhs.vertices != rhs.vertices) || (lhs.neighbors != rhs.neighbors);
}
static bool operator!=(const TriangleTopology::BoundaryInfo& lhs, const TriangleTopology::BoundaryInfo& rhs) {
  return (lhs.prev != rhs.prev) 
      || (lhs.next != rhs.next)
      || (lhs.reverse != rhs.reverse)
      || (lhs.src != rhs.src);
}

static bool modified_mesh(const MeshAndX& mesh_and_x1,
                          const MeshAndX& mesh_and_x2) {
  const TriangleTopology& m1 = *mesh_and_x1.x;
  const TriangleTopology& m2 = *mesh_and_x2.x;
  if(m1.faces_.flat != m2.faces_.flat) return true;
  if(m1.vertex_to_edge_.flat != m2.vertex_to_edge_.flat) return true;
  if(m1.boundaries_ != m2.boundaries_) return true;
  return false;
}

static T mesh_volume(const MeshAndX& mesh_and_x) {
  const auto& mesh = *mesh_and_x.x;
  const auto& X = mesh_and_x.y;
  GEODE_ASSERT(!mesh.has_boundary());
  T sum = 0;
  for(const FaceId f : mesh.faces()) {
    const auto verts = mesh.vertices(f);
    sum += det(X[verts[0]],X[verts[1]],X[verts[2]]);
  }
  return T(1./6)*sum;
}

void test_simplify_case(Tuple<Ref<TriangleSoup>,Array<Vector<real,3>>> soup_and_x) {
  const auto mesh = new_<TriangleTopology>(soup_and_x.x);
  const auto X = Field<TV,VertexId>{soup_and_x.y.copy()};
  const auto input = MeshAndX{mesh,X};

  {
    // If every point in mesh is degenerate, simplify should eliminate everything 
    const auto zero_X = Field<TV,VertexId>{X.size()};
    const auto simplified = simplify(mesh,zero_X,0.);
    GEODE_ASSERT(simplified.x->n_faces() == 0);
    GEODE_ASSERT(simplified.x->n_vertices() == 0);
  }

  {
    const auto simplified = simplify(mesh,X,0.);
    simplified.x->assert_consistent();
    GEODE_ASSERT(input.x->chi() <= simplified.x->chi()); // This should only ever be increased
    GEODE_ASSERT(abs(mesh_volume(input) - mesh_volume(simplified)) <= 1e-6); // Volume shouldn't be significantly different
    GEODE_ASSERT(count_degenerate_edges(simplified, 0.) == 0); // All degenerate edges should be removed
    GEODE_ASSERT(count_unconnected_clusters(simplified, 0.) == 0); // Adjacent vertices should have been merged (TODO: Implement additional operations in simplify to do this)
    const auto simplified2 = simplify(simplified.x,simplified.y,0.);
    GEODE_ASSERT(!modified_mesh(simplified, simplified2)); // After simplification, further simplification should do nothing
  }
}

void test_simplify_helper() {
  // Empty mesh
  test_simplify_case(tuple(new_<TriangleSoup>(Array<const Vector<int,3>>{}),Array<Vector<real,3>>{}));
  // Tetrahedron
  test_simplify_case(tetrahedron_mesh());
}

}
using namespace geode;

void wrap_decimate() {
  GEODE_FUNCTION(decimate)
  GEODE_FUNCTION(decimate_inplace)
  GEODE_FUNCTION(simplify)
  GEODE_FUNCTION_2(simplify_inplace, simplify_inplace_python)
  GEODE_FUNCTION(test_simplify_helper)

}
