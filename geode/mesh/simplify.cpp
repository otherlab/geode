#include <geode/mesh/quadric.h>
#include <geode/mesh/simplify.h>
#include <geode/structure/Heap.h>

namespace geode {
typedef real T;
typedef Vector<T,3> TV;

// We perform a series of collapse and flip operations intended to remove degenerate geometry from a mesh
// A Heap of candidate simplifications is maintained to ensure the cheapest operations are performed first
// These are the simplifications (in order of priority) that are considered for each edge
// 1.) Collapse all zero length edges (splitting loops as needed to fill/cut degenerate holes/handles)
// 2.) Collapse all remaining non-zero length edges within allowed error bounds (splitting as before)
// 3.) Erase any degenerate face pairs
// 4.) Flip any edges that enable an additional collapse
// 5.) Flip any edges that reduce number of degenerate faces
// Within a priority, the lowest cost edge is simplified first
// All but the last step will remove at least one vertex or decrease the topological complexity of the mesh ensuring algorithm will eventually terminate
// The last step should ensure that checks for degenerate faces in a way that should ensure faces will be consistently detected as degenerate or not

// TODO: This implementation defers some checks that determine if a simplification is valid until that simplification has the lowest cost
//   This is likely not optimal for large meshes since inserting operation onto heap is O(log(n)) while those checks should be linear or quadratic in vertex degree (which will average to 6 for most meshes)
//   However, I think performing those checks earlier requires invalidating more pending simplifications when changes occur and those checks likely have a relatively large constant cost
//   Until I can perform some benchmarks on real data sets, I'm leaving things as-is

// TODO: There is a bunch of nearly identical code in decimate.cpp that should be de-duplicated or combined for easier maintenance
//   That should be easier when I clean up the old version of simplify, but I'm saving both for a few commits to compare performance
namespace {

// Compute quadric for a single face (handling degenerate cases appropriately)
static Quadric face_quadric(const MutableTriangleTopology& mesh, const RawField<const TV, VertexId> X, const FaceId f) {
  Quadric result;
  const auto total = result.add_face(mesh, X, f);
  if (total) {
    const real inv_total = 1/total;
    result.A *= inv_total;
    result.b *= inv_total;
    result.c *= inv_total;
  }
  return result;
}

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


struct SimplificationCandidate {

  enum class Step {
    collapse_zero_length_edge,
    collapse_edge,
    erase_face_pair,
    check_flip_and_collapse,
    flip_and_collapse,
    check_flip,
    flip_edge
  };
  HalfedgeId he;
  Step step;
  T cost;
  inline friend bool operator<(const SimplificationCandidate& lhs, const SimplificationCandidate& rhs) {
    if(lhs.step != rhs.step) return lhs.step < rhs.step;
    if(lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
    return lhs.he < rhs.he;
  }

  static SimplificationCandidate collapse_zero_length_edge(const HalfedgeId he)
  { return {he, Step::collapse_zero_length_edge, 0.}; }
  static SimplificationCandidate collapse_edge(const HalfedgeId he, const T cost)
  { return {he, Step::collapse_edge, cost}; }
  static SimplificationCandidate erase_face_pair(const HalfedgeId he)
  { return {he, Step::erase_face_pair, 0.}; } // Might be better to erase smaller area faces first, but I don't think order matters
  static SimplificationCandidate check_flip_and_collapse(const HalfedgeId he, const T cost)
  { return {he, Step::check_flip_and_collapse, cost}; }
  static SimplificationCandidate flip_and_collapse(const HalfedgeId he, const T cost)
  { return {he, Step::flip_and_collapse, cost}; }
  static SimplificationCandidate check_flip(const HalfedgeId he, const T cost)
  { return {he, Step::check_flip, cost}; }
  static SimplificationCandidate flip_edge(const HalfedgeId he, const T cost)
  { return {he, Step::flip_edge, cost}; }

};

static std::string str(SimplificationCandidate::Step s) {
  switch(s) {
    case SimplificationCandidate::Step::collapse_zero_length_edge: return "collapse_zero_length_edge";
    case SimplificationCandidate::Step::collapse_edge: return "collapse_edge";
    case SimplificationCandidate::Step::erase_face_pair: return "erase_face_pair";
    case SimplificationCandidate::Step::check_flip_and_collapse: return "check_flip_and_collapse";
    case SimplificationCandidate::Step::flip_and_collapse: return "flip_and_collapse";
    case SimplificationCandidate::Step::check_flip: return "check_flip";
    case SimplificationCandidate::Step::flip_edge: return "flip_edge";
  }
  return "<Invalid SimplificationCandidate::Step>";
}

static std::ostream& operator<<(std::ostream& os, const SimplificationCandidate::Step s) {
  return os << str(s);
}

// Field that can handle interior and exterior halfedges from a TriangleTopology
template<class T> struct HalfedgeField {
  Array<T> boundary_flat;
  Array<T> interior_flat;

  HalfedgeField()=default;
  HalfedgeField(const TriangleTopology& mesh, const T default_value)
  : boundary_flat(mesh.boundaries_.size(), uninit)
  , interior_flat(mesh.allocated_halfedges(), uninit)
  {
    boundary_flat.fill(default_value);
    interior_flat.fill(default_value);
  }

  int size() const { return boundary_flat.size() + interior_flat.size(); }
  bool empty() const { return interior_flat.empty() && boundary_flat.empty(); }

  static bool is_interior(const HalfedgeId he)
  { return (he.idx() >= 0); }

  static int boundary_index(const HalfedgeId he) {
    assert(!is_interior(he));
    return -1 - he.idx();
  }

  static HalfedgeId boundary_id(const int index)
  { return HalfedgeId{-1 - index}; }

  HalfedgeId lo_id() const { return boundary_id(boundary_flat.size()-1); }
  HalfedgeId hi_id() const { return HalfedgeId(interior_flat.size()); }

  Range<IdIter<HalfedgeId>> id_range() const { return geode::id_range(lo_id(), hi_id()); }

  bool valid(const HalfedgeId he) const { return is_interior(he) ? interior_flat.valid(he.idx())
                                                                 : boundary_flat.valid(boundary_index(he)); }

  void fill(const T x)
  { interior_flat.fill(x); boundary_flat.fill(x); }

  T& operator[](const HalfedgeId he) {
    return is_interior(he) ? interior_flat[he.idx()]
                           : boundary_flat[boundary_index(he)];
  }

  const T& operator[](const HalfedgeId he) const {
    return is_interior(he) ? interior_flat[he.idx()]
                           : boundary_flat[boundary_index(he)];
  }
};

template<class T> static void grow_to_fit(Array<T>& a, const int index, const T default_value) {
  const int old_size = a.size();
  if(index >= old_size) {
    const int new_size = index + 1;
    a.resize(new_size, uninit);
    a.slice(old_size, new_size).fill(default_value);
  }
}

template<class T, class Id> static void grow_to_fit(Field<T, Id>& f, const Id id, const T default_value)
{ grow_to_fit(f.flat, id.idx(), default_value); }

template<class T> static void grow_to_fit(HalfedgeField<T>& f, const HalfedgeId he, const T default_value)
{
    if(f.is_interior(he)) {
      grow_to_fit(f.interior_flat,he.idx(),default_value);
    }
    else {
      grow_to_fit(f.boundary_flat,f.boundary_index(he),default_value);
    }
}

struct SimplifyHeap : public HeapBase<SimplifyHeap>, public Noncopyable {
  typedef HeapBase<SimplifyHeap> Base;
  Array<SimplificationCandidate> heap;
  HalfedgeField<int> inv_heap;

  SimplifyHeap(const TriangleTopology& mesh)
   : inv_heap(mesh,-1)
  { }

  int size() const { return heap.size(); }
  bool empty() const { return size() == 0; }
  bool contains(const HalfedgeId he) { return inv_heap.valid(he) && inv_heap[he] != -1; }
  bool first(const int i, const int j) const {
    return heap[i] < heap[j];
  }

  void swap(const int i, const int j) {
    std::swap(heap[i],heap[j]);
    inv_heap[heap[i].he] = i;
    inv_heap[heap[j].he] = j;
  }

  SimplificationCandidate pop() {
    const auto next = heap[0];
    inv_heap[next.he] = -1;
    const auto p = heap.pop();
    if (size()) {
      heap[0] = p;
      inv_heap[heap[0].he] = 0;
      Base::move_downward(0);
    }
    return next;
  }

  void set(const SimplificationCandidate candidate) {
    const auto he = candidate.he;
    // Grow heap as needed if id is out of range
    grow_to_fit(inv_heap, he, -1);
    int& i = inv_heap[he];
    if (i < 0)
      i = heap.append(candidate);
    else
      heap[i] = candidate;
    Base::move_up_or_down(i);
  }

  void erase(const HalfedgeId he) {
    if(!inv_heap.valid(he))
      return; // Value was never set so we can ignore it
    int& i = inv_heap[he];
    if (i >= 0) {
      const auto p = heap.pop();
      if (i < size()) {
        heap[i] = p;
        inv_heap[p.he] = i;
        Base::move_up_or_down(i);
      }
      i = -1;
    }
  }
};

// Provide fast insertion of a specific id and fast removal of some 'next' id
template<class Id,class FieldT=Field<bool,Id>> struct DirtySet {
  Array<Id> dirty_list;
  FieldT dirty_flags;
  void append_unique(const Id id) {
    grow_to_fit(dirty_flags, id, false);
    if(!dirty_flags[id]) {
      dirty_flags[id] = true;
      assert(!dirty_list.contains(id));
      dirty_list.append(id);
    }
  }
  bool empty() const { return dirty_list.empty(); }
  int size() const { return dirty_list.size(); }
  Id pop() {
    assert(!dirty_list.empty());
    Id result = dirty_list.pop();
    if(dirty_flags.valid(result)) {
      dirty_flags[result] = false;
    }
    return result;
  }
  bool contains(const Id id) const
  { return dirty_flags.valid(id) && dirty_flags[id]; }
  auto begin() const -> decltype(dirty_list.begin()) { return dirty_list.begin(); }
  auto end() const -> decltype(dirty_list.end()) { return dirty_list.end(); }
};

// Simplification steps involve several of the same checks and operations used in slightly different orders; most of these need many of the same arguments
// I've moved all of the common data into a struct to avoid passing the same arguments around all over the place
struct SimplifyHelper {
  MutableTriangleTopology& mesh;
  const Field<TV,VertexId>& X;
  const T distance_eps;
  const T max_angle;
  const int min_vertices;
  const T boundary_distance;
  const T area_eps = sqr(distance_eps);
  // We consider face to be degenerate if face area is smaller than sqr(distance) (this is a mostly arbitrary choice, but at least the units match)
  // area_of_face = 0.5*magnitude(n) < sqr(distance)
  // sqr_magnitude(n) < 4.*sqr(sqr(distance))
  const T sqr_magnitude_n_eps = 4.*sqr(area_eps); // Threshold for testing if faces are degenerate
  const T sign_sqr_min_cos = sign_sqr(max_angle > .99*pi ? -1 : cos(max_angle));

  // Quadrics for each vertex of mesh
  Field<Quadric,VertexId> cached_quadrics{mesh.allocated_vertices()};
  // This is a heap of the next candidate simplification for each halfedge
  // Need to be careful that an invalid simplification with a lower cost doesn't hide a valid one
  SimplifyHeap heap{mesh};
  // Set of vertices where quadric needs to be updated
  DirtySet<VertexId> dirty_quadrics;
  // Set of halfedges where simplification operation needs to be updated
  DirtySet<HalfedgeId,HalfedgeField<bool>> dirty_edges;

  const Quadric& quadric(const VertexId v) const {
    assert(!dirty_quadrics.contains(v));
    return cached_quadrics[v];
  }

  SimplifyHelper(MutableTriangleTopology& _mesh, const Field<TV,VertexId>& _X,
    const T _distance_eps, const T _max_angle, const int _min_vertices, const T _boundary_distance)
   : mesh(_mesh)
   , X(_X)
   , distance_eps(_distance_eps)
   , max_angle(_max_angle)
   , min_vertices(_min_vertices)
   , boundary_distance(_boundary_distance)
  {
    mesh.erase_isolated_vertices();

    // Start with an updated quadric for each vertex
    for(const VertexId v : mesh.vertices()) {
      cached_quadrics[v] = compute_quadric(mesh, X, v);
    }

    // Compute the initial cost to simplify each edge
    for(const HalfedgeId he : mesh.halfedges()) {
      set_simplify_cost_initial(he);
    }
  }


  // Check if a triangle between 3 vertices should count as degenerate
  // Result should remain consistent even if order of vertices is rotated
  bool is_triangle_degenerate(const VertexId v0, const VertexId v1, const VertexId v2) const {
    const auto unsorted_ids = Vector<VertexId,3>{v0,v1,v2};
    const auto canonical_ids = unsorted_ids.roll(unsorted_ids.argmin());
    assert(canonical_ids.argmin() == 0);
    const auto x0 = X[canonical_ids[0]];
    const auto x1 = X[canonical_ids[1]];
    const auto x2 = X[canonical_ids[2]];
    const auto n = cross(x2-x1,x0-x1);
    return sqr_magnitude(n) <= sqr_magnitude_n_eps;
  }

  bool collapse_changes_normal_too_much(const HalfedgeId he) const {
    if (sign_sqr_min_cos > -1) {
      const VertexId vs = mesh.src(he);
      const VertexId vd = mesh.dst(he);
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
        if (he!=ee && !mesh.is_boundary(ee)) {
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
  }

  bool collapse_moves_boundary_too_much(const HalfedgeId he) const {
    const HalfedgeId b = mesh.halfedge(mesh.src(he)); // Boundary halfedge if src is on boundary
    if(!mesh.is_boundary(b))
      return false; // Not a boundary edge
    const auto xd = X[mesh.dst(he)];
    return (line_point_distance(mesh.segment(X, b),xd) > boundary_distance
         || line_point_distance(mesh.segment(X, mesh.prev(b)),xd) > boundary_distance);
  }

  void set_simplify_cost_initial(const HalfedgeId he) {
    assert(!heap.contains(he));
    const VertexId vs = mesh.src(he);
    const VertexId vd = mesh.dst(he);
    if(X[vs] == X[vd]) {
      // This is a zero length edge which won't move vertex positions making it easier to handle
      heap.set(SimplificationCandidate::collapse_zero_length_edge(he));
      return;
    }
    const T cost = quadric(vs)(X[vd]);
    if(cost <= area_eps) {
      // This edge can be collapsed... probably
      // Try to collapse it first, and add it back to queue for further considerations if not
      heap.set(SimplificationCandidate::collapse_edge(he, cost));
      return;
    }
    // If we ruled out a collapse, try a flip
    set_simplify_cost_post_collapse(he);
  }

  // Queue appropriate simplification options for an edge that can't be collapsed
  void set_simplify_cost_post_collapse(const HalfedgeId he) {
    if(!mesh.is_boundary(he) && !mesh.is_boundary(mesh.reverse(he))) {
      const auto rev_he = mesh.reverse(he);
      const auto vl = mesh.opposite(he);
      const auto vr = mesh.opposite(rev_he);
      if(vl == vr) {
        // This edge is a crease between two faces that are reversed copies of each other
        heap.set(SimplificationCandidate::erase_face_pair(he));
        return;
      }

      // All further operations start with an edge flip
      if(!mesh.is_flip_safe(he)) {
        return;
      }

      // Collapses should remove all degenerate edges, but we can still have degenerate triangles that are almost lines
      // There are different possibilities for how to remove those
      // We can't directly collapse any of the edges, but we might be able to collapse a vertex to the opposite segment
      // This looks like inserting a new vertex on the edge and collapsing the opposite vertex to it (if this doesn't violate error bounds)
      // The net effect of this is an edge flip which we can perform directly without introducing an intermediate vertex
      const auto old_seg = mesh.segment(X, he);
      const auto new_seg = Segment<TV>{X[vl], X[vr]}; 

      // We look at separation between segments to check for error
      const auto closest_approach = segment_segment_distance_and_normal(old_seg, new_seg); // Returns distance, normal, and weights
      if(closest_approach.x <= distance_eps // Need to make sure new_seg isn't too far from old_seg
        && closest_approach.z.x != 0. // If we are collapsing to an endpoint we would have done so already...
        && closest_approach.z.x != 1. // ...also if closest point is at start or end of old_seg, we are probably dealing with a concave quad
        ) {
        // Quadric for new point is just the sum from the two face before splitting
        // When we collapse new point to vl, we are on that face so don't need to compute that one
        // We just need to compute error from opposite face
        const auto flip_cost = face_quadric(mesh, X, mesh.face(rev_he))(X[vl]);
        if(flip_cost <= area_eps) {
          // There's additional checks needed before we actually do the flip, but we defer those until we run out of better simplification steps
          heap.set(SimplificationCandidate::check_flip_and_collapse(he, flip_cost));
        }
      }
    }
  }

  void set_simplify_cost_post_flip(const HalfedgeId he, const T flip_cost) {
    // After we flip this edge will we be able to collapse the resulting edge (best) or at least have fewer degenerate faces?
    assert(mesh.is_flip_safe(he)); // Should have already checked this
    const HalfedgeId rev_he = mesh.reverse(he);
    const VertexId vs = mesh.src(he);
    const VertexId vd = mesh.dst(he);
    const VertexId vl = mesh.opposite(he);
    const VertexId vr = mesh.opposite(rev_he);
    const HalfedgeId pe = mesh.prev(he);

    // Compute quadric for vl if we flipped he
    T total = 0;
    Quadric ql;
    for(const HalfedgeId h : mesh.outgoing(vl)) {
      if(h == pe) continue; // Skip face that will be split after flip
      if(mesh.is_boundary(h)) continue; // Skip boundary since no face
      total += ql.add_face(mesh, X, mesh.face(h));
    }
    const auto add_plane = [&](const TV x0, const TV x1, const TV x2) {
      total += ql.add_plane(x0, cross(x1-x0, x2-x0));
    };
    const auto xl = X[vl];
    const auto xr = X[vr];
    const auto xs = X[vs];
    const auto xd = X[vd];
    add_plane(xl, xs, xr);
    add_plane(xl, xr, xd);
    if(total) {
      ql *= 1/total;
    }
    const auto collapse_cost = ql(xr);
    if(collapse_cost <= area_eps) {
      // After flipping this edge we might be able to collapse it
      heap.set(SimplificationCandidate::flip_and_collapse(he, flip_cost + collapse_cost));
    }
    else {
      set_simplify_cost_just_flip(he, flip_cost);
    }
  }

  void set_simplify_cost_just_flip(const HalfedgeId he, const T flip_cost) {
    // Can't collapse, but flipping still might reduce number of degenerate faces
    // Need to be very careful that we are consistent about how we count degenerate faces to avoid an infinite loop
    const HalfedgeId rev_he = mesh.reverse(he);
    const VertexId vs = mesh.src(he);
    const VertexId vd = mesh.src(rev_he);
    const VertexId vl = mesh.opposite(he);
    const VertexId vr = mesh.opposite(rev_he);
    const int num_degenerate_before = static_cast<int>(is_triangle_degenerate(vs, vd, vl))
                                    + static_cast<int>(is_triangle_degenerate(vd, vs, vr));
    const int num_degenerate_after = static_cast<int>(is_triangle_degenerate(vl, vr, vd))
                                   + static_cast<int>(is_triangle_degenerate(vr, vl, vs));
    if(num_degenerate_after < num_degenerate_before) {
      heap.set(SimplificationCandidate::flip_edge(he, flip_cost));
    }
  }

  void set_simplify_cost_just_flip(const HalfedgeId he) {
    const VertexId vl = mesh.opposite(he);
    const auto flip_cost = face_quadric(mesh, X, mesh.face(mesh.reverse(he)))(X[vl]);
    if(flip_cost <= area_eps) {
      set_simplify_cost_just_flip(he, flip_cost);
    }
  }

  // Collapse an edge of the mesh if safe, or split
  // Marks affected quadrics and edges as dirty, but doesn't update them on heap unless they are erased
  void do_collapse_or_split(const HalfedgeId he) {
    if(mesh.is_collapse_safe(he)) {
      const VertexId vs = mesh.src(he);
      const VertexId vd = mesh.dst(he);
      // Edges in collapsed faces are all about to be erased. Clear them from the heap
      assert(!heap.contains(he)); // Collapsed edge should have been popped from heap before calling this function
      if(!mesh.is_boundary(he)) {
        heap.erase(mesh.next(he));
        heap.erase(mesh.prev(he));
      }
      const auto rev_he = mesh.reverse(he);
      heap.erase(rev_he);
      if(!mesh.is_boundary(rev_he)) {
        heap.erase(mesh.next(rev_he));
        heap.erase(mesh.prev(rev_he));
      }
      // Collapse will replace X[vs] with X[vd]
      // Neighboring vertices of X[vs] will need to update quadrics (unless X[vs] was already equal to X[vd] before collapse)
      // When position of a vertex changes, incoming edges need to reevaluate collapse cost
      // When quadric at a vertex changes, outgoing edges need to reevaluate collapse cost
      // Outgoing edges from those vertices
      // 
      // cost = quadrics[vs](X[vd])
      // Unless X[vs] == X[vd], edges ending at vs will need to reevaluate cost
      if(X[vs] != X[vd]) {
        // Walk the fan of edges around vs that aren't about to be erased and mark them as dirty
        for(HalfedgeId out_he = mesh.left(mesh.left(he)); out_he != he; out_he = mesh.left(out_he)) {
          // We mark incoming edge here (Outgoing edges only need to be updated if quadric changes which is covered later)
          dirty_edges.append_unique(mesh.reverse(out_he));
        }
      }

      // Do the actual collapse
      mesh.unsafe_collapse(he);

      // Mark all affected quadrics for updating
      dirty_quadrics.append_unique(vd);
      for(const HalfedgeId out_he : mesh.outgoing(vd)) {
        dirty_quadrics.append_unique(mesh.dst(out_he));
      }
    }
    else {
      // There should be an intersection in one-rings of vs and vd that are preventing the collapse
      const auto loop = find_loop(mesh, he);
      if(loop.x == he) {
        // We can fix this by splitting the loop of edges
        // If there are multiple intersections, we might have to repeat this multiple times before we can collapse
        const auto new_faces = mesh.split_loop(loop.x, loop.y, loop.z);

        // Mark affected changes
        for(const FaceId f : new_faces) {
          for(const VertexId v : mesh.vertices(f)) {
            dirty_quadrics.append_unique(v);
          }
          for(const HalfedgeId he : mesh.halfedges(f)) {
            dirty_edges.append_unique(he);
            dirty_edges.append_unique(mesh.reverse(he));
          }
        }
      }
      else {
        // If we don't find a loop, check if we have a pair of degenerate faces
        if(!mesh.is_boundary(he) && !mesh.is_boundary(mesh.reverse(he)) 
          && mesh.opposite(he) == mesh.opposite(mesh.reverse(he))) {
          // This component of a mesh has been reduced to a pair of degenerate faces
          // We erase those faces
          do_erase_face_pair(he);
        }
        else {
          // If we get here, something complicated is going on. Probably related to boundary
          // We should be able to ignore it
        }
      }
    }
  }

  void do_erase_face_pair(const HalfedgeId he) {
    // Collapse turned a tetrahedron into a pair of degenerate faces
    assert(mesh.opposite(he) == mesh.opposite(mesh.reverse(he)));

    // Mark affected quadrics
    dirty_quadrics.append_unique(mesh.src(he));
    dirty_quadrics.append_unique(mesh.dst(he));
    dirty_quadrics.append_unique(mesh.opposite(he));

    // Clear any edges that will be erased from heap
    // Mark adjacent edges as dirty
    for(const FaceId f : mesh.faces(he)) {
      for(const HalfedgeId h2 : mesh.halfedges(f)) {
        heap.erase(h2);
        dirty_edges.append_unique(mesh.reverse(h2));
      }
    }

    mesh.collapse_degenerate_face_pair(he);
  }

  void do_flip_and_collapse(const HalfedgeId old_he) {
    assert(mesh.is_flip_safe(old_he));
    const auto undo = mesh.save_state_before_flip(old_he);
    const HalfedgeId new_he = mesh.unsafe_flip_edge(old_he);
    if(collapse_changes_normal_too_much(new_he) || collapse_moves_boundary_too_much(new_he)) {
      mesh.unflip_edge(undo);
      set_simplify_cost_just_flip(old_he);
    }
    else {
      // We don't need to mark anything as dirty since the collapse afterwards touches everything that would be affected
      assert(!collapse_changes_normal_too_much(new_he));
      assert(!collapse_moves_boundary_too_much(new_he));
      assert(mesh.is_collapse_safe(new_he));
      do_collapse_or_split(new_he);
    }
  }

  void do_flip(const HalfedgeId old_he) {
    assert(mesh.is_flip_safe(old_he));
    const HalfedgeId new_he = mesh.unsafe_flip_edge(old_he);
    for(const FaceId f : mesh.faces(new_he)) {
      for(const HalfedgeId he : mesh.halfedges(f)) {
        dirty_edges.append_unique(he);
      }
    }
    for(const VertexId v : {mesh.src(new_he), mesh.dst(new_he), mesh.opposite(new_he), mesh.opposite(mesh.reverse(new_he))}) {
      dirty_quadrics.append_unique(v);
    }
  }

  // Updates dirty quadrics and simplification option
  void update_dirty_stuff() {
    while(!dirty_quadrics.empty()) {
      const VertexId v = dirty_quadrics.pop();
      if(mesh.erased(v)) continue;
      const auto new_quadric = compute_quadric(mesh, X, v);
      if(cached_quadrics.valid(v) && !(cached_quadrics[v] != new_quadric)) {
        // Quadric didn't change so don't need to update edges that depend on it
        continue;
      }
      if(!cached_quadrics.valid(v)) {
        cached_quadrics.flat.resize(v.idx()+1); // Grow for a new vertex
      }
      cached_quadrics[v] = new_quadric;
      for(const HalfedgeId he : mesh.outgoing(v)) {
        dirty_edges.append_unique(he);
      }
    }
    for(const HalfedgeId he : dirty_edges) {
      heap.erase(he);
    }
    while(!dirty_edges.empty()) {
      const HalfedgeId he = dirty_edges.pop();
      if(!mesh.erased(he)) {
        set_simplify_cost_initial(he);
      }
    }
  }

  void run() {
    for(;;) {
      if(mesh.n_vertices() <= min_vertices)
        break;
      update_dirty_stuff();
      if(heap.empty())
        break;
      const SimplificationCandidate next = heap.pop();
      const HalfedgeId he = next.he;
      switch(next.step) {
        case SimplificationCandidate::Step::collapse_zero_length_edge:
          assert(X[mesh.src(he)] == X[mesh.dst(he)]); // Should be exactly zero length edge
          assert(!collapse_changes_normal_too_much(he)); // Vertex positions should be unchanged so should always be ok
          assert(!collapse_moves_boundary_too_much(he)); // Don't need to check boundary distance since vertex isn't actually moving
          do_collapse_or_split(he);
          break;
        case SimplificationCandidate::Step::collapse_edge:
          if(    !collapse_changes_normal_too_much(he)
              && !collapse_moves_boundary_too_much(he)) {
            do_collapse_or_split(he);
          }
          else {
            // Collapsing this edge violates error bounds
            // We queue advanced simplification steps
            set_simplify_cost_post_collapse(he);
          }
          break;
        case SimplificationCandidate::Step::erase_face_pair:
          do_erase_face_pair(he);
          break;
        case SimplificationCandidate::Step::check_flip_and_collapse:
          set_simplify_cost_post_flip(he, next.cost);
          break;
        case SimplificationCandidate::Step::flip_and_collapse:
          do_flip_and_collapse(he);
          break;
        case SimplificationCandidate::Step::check_flip:
          set_simplify_cost_just_flip(he);
          break;
        case SimplificationCandidate::Step::flip_edge:
          do_flip(he);
          break;
      }
    }
  }
};
} // anonymous namespace

void simplify_inplace(MutableTriangleTopology& mesh,
                 const FieldId<Vector<real,3>,VertexId> X_id, // X must be a field managed by the mesh so that it will be properly updated if topological simplification creates new vertices
                 const real distance,             // (Very) approximate distance between original and decimation
                 const real max_angle,       // Max normal angle change in radians for one decimation step
                 const int min_vertices,       // Stop if we decimate down to this many vertices (-1 for no limit)
                 const real boundary_distance) {

  auto&& helper = SimplifyHelper{mesh, mesh.field(X_id), distance, max_angle, min_vertices, boundary_distance};
  helper.run();
}

Tuple<Ref<const TriangleTopology>,Field<const Vector<real,3>,VertexId>>
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

} // geode namespace