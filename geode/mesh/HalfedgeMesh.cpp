// A halfedge data structure representing oriented triangle meshes.

#include <geode/mesh/HalfedgeMesh.h>
#include <geode/utility/str.h>
#include <geode/array/Nested.h>
#include <geode/python/Class.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/utility/Log.h>
#include <geode/vector/convert.h>
#include <boost/dynamic_bitset.hpp>
namespace geode {

using Log::cout;
using std::endl;

GEODE_DEFINE_TYPE(HalfedgeMesh)

HalfedgeMesh::HalfedgeMesh() {}

HalfedgeMesh::HalfedgeMesh(const HalfedgeMesh& mesh)
  : halfedges_(mesh.halfedges_.copy())
  , vertex_to_edge_(mesh.vertex_to_edge_.copy())
  , face_to_edge_(mesh.face_to_edge_.copy()) {}

HalfedgeMesh::~HalfedgeMesh() {}

Ref<HalfedgeMesh> HalfedgeMesh::copy() const {
  return new_<HalfedgeMesh>(*this);
}

HalfedgeId HalfedgeMesh::halfedge(VertexId v0, VertexId v1) const {
  const auto start = halfedge(v0);
  if (start.valid()) {
    auto e = start;
    do {
      if (dst(e)==v1)
        return e;
      e = left(e);
    } while (e != start);
  }
  return HalfedgeId();
}

HalfedgeId HalfedgeMesh::common_halfedge(FaceId f0, FaceId f1) const {
  if (f0.valid()) {
    for (const auto e : halfedges(f0))
      if (face(reverse(e))==f1)
        return e;
  } else if (f1.valid()) {
    for (const auto e : halfedges(f1))
      if (face(reverse(e))==f0)
        return reverse(e);
  }
  return HalfedgeId();
}

VertexId HalfedgeMesh::add_vertex() {
  return vertex_to_edge_.append(HalfedgeId());
}

void HalfedgeMesh::add_vertices(int n) {
  vertex_to_edge_.flat.resize(vertex_to_edge_.size()+n);
}

static inline HalfedgeId right_around_dst_to_boundary(const HalfedgeMesh& mesh, HalfedgeId e) {
  e = mesh.reverse(mesh.next(mesh.reverse(e)));
  while (!mesh.is_boundary(e))
    e = mesh.reverse(mesh.next(e));
  return e;
}

// Set halfedge(v) to an outgoing boundary halfedge if possible, or any halfedge otherwise.  We start
// looping at the given initial guess, which should ideally be a likely boundary edge.
static inline void fix_vertex_to_edge(HalfedgeMesh& mesh, const VertexId v, const HalfedgeId start) {
  auto e = start;
  for (;;) {
    if (mesh.is_boundary(e))
      break;
    e = mesh.left(e);
    if (e==start)
      break;
  }
  mesh.vertex_to_edge_[v] = e;
}

GEODE_COLD static void add_face_error(const Vector<VertexId,3> v, const char* reason) {
  throw RuntimeError(format("HalfedgeMesh::add_face: can't add face (%d,%d,%d)%s",v.x.idx(),v.y.idx(),v.z.idx(),reason));
}

FaceId HalfedgeMesh::add_face(const Vector<VertexId,3> v) {
  // Check for errors
  if (!valid(v.x) || !valid(v.y) || !valid(v.z))
    add_face_error(v," containing invalid vertex");
  if (v.x==v.y || v.y==v.z || v.z==v.x)
    add_face_error(v," containing duplicate vertex");
  if (!is_boundary(v.x) || !is_boundary(v.y) || !is_boundary(v.z))
    add_face_error(v,", one of the vertices is interior");
  auto e0 = halfedge(v.x,v.y),
       e1 = halfedge(v.y,v.z),
       e2 = halfedge(v.z,v.x);
  if (   (e0.valid() && !is_boundary(e0))
      || (e1.valid() && !is_boundary(e1))
      || (e2.valid() && !is_boundary(e2)))
    add_face_error(v,", one of the edges is interior");

  // If a vertex has multiple triangle fan boundaries, there is ambiguity in how the boundary halfedges are linked
  // together into components.  This arbitrary choice may be incompatible with the new face, in which case surgery
  // is required to correct the linkage.  We check for all errors before we do any actual surgery so that add_face
  // makes changes only when it succeeds.
  #define PREPARE(a,b,c) \
    const auto c = a.valid() && b.valid() && next(a)!=b ? right_around_dst_to_boundary(*this,b) : HalfedgeId(); \
    if (c.valid() && b==c) \
      add_face_error(v,", result would consist of a closed ring plus extra triangles");
  PREPARE(e0,e1,c0)
  PREPARE(e1,e2,c1)
  PREPARE(e2,e0,c2)
  // If we get to this point, all checks have passed, and we can safely add the new face.  For each pair of
  // adjacent ei,ej, we want next(ei)==ej.  We've already computed ci = the other side of ej's triangle fan.
  // Here is the surgery:
  #define RELINK(a,b,c) \
    if (c.valid()) { \
      const auto na = next(a), pb = prev(b), nc = next(c); \
      unsafe_link(a,b); \
      unsafe_link(c,na); \
      unsafe_link(pb,nc); \
    }
  RELINK(e0,e1,c0)
  RELINK(e1,e2,c1)
  RELINK(e2,e0,c2)

  // Create missing edges
  const int new_edges = 2*(3-e0.valid()-e1.valid()-e2.valid());
  if (new_edges) {
    int e_next = halfedges_.flat.extend(new_edges,uninit);
    #define ENSURE_EDGE(e,ep,en,vs,vd,ep_always,en_always) \
      if (!e.valid()) { \
        e = HalfedgeId(e_next++); \
        const auto r = HalfedgeId(e_next++); \
        halfedges_[e].src = vs; \
        halfedges_[r].src = vd; \
        halfedges_[r].face = FaceId(); \
        /* Set links around vs */ \
        if (ep_always || ep.valid()) { \
          const auto n = next(ep); \
          unsafe_link(ep,e); \
          unsafe_link(r,n); \
        } else { \
          const auto n = halfedge(vs); \
          if (n.valid()) { \
            const auto p = prev(n); \
            unsafe_link(p,e); \
            unsafe_link(r,n); \
          } else \
            unsafe_link(r,e); \
        } \
        /* Set links around vd */ \
        if (en_always || en.valid()) { \
          const auto p = prev(en); \
          unsafe_link(e,en); \
          unsafe_link(p,r); \
        } else { \
          const auto n = halfedge(vd); \
          if (n.valid()) { \
            const auto p = prev(n); \
            unsafe_link(e,n); \
            unsafe_link(p,r); \
          } else \
            unsafe_link(e,r); \
        } \
      }
    // Ensure that each edge exists, assuming that edges we've already checked exist.
    ENSURE_EDGE(e0,e2,e1,v.x,v.y,false,false)
    ENSURE_EDGE(e1,e0,e2,v.y,v.z,true,false)
    ENSURE_EDGE(e2,e1,e0,v.z,v.x,true,true)
  }

  // Fix vertex to edge pointers to point to boundary halfedges if possible
  fix_vertex_to_edge(*this,v.x,reverse(e2));
  fix_vertex_to_edge(*this,v.y,reverse(e0));
  fix_vertex_to_edge(*this,v.z,reverse(e1));

  // Create the new face!  Victory at last!
  const auto f = face_to_edge_.append(e0);
  halfedges_[e0].face = halfedges_[e1].face = halfedges_[e2].face = f;
  return f;
}

void HalfedgeMesh::add_faces(RawArray<const Vector<int,3>> vs) {
  // TODO: Write a faster batch insertion routine
  for (auto& v : vs)
    add_face(Vector<VertexId,3>(v));
}

void HalfedgeMesh::split_face(const FaceId f, const VertexId c) {
  GEODE_ASSERT(isolated(c));
  const auto e = halfedges(f);
  const auto v = vertices(f);
  const int f_base = face_to_edge_.flat.extend(2,uninit);
  const int e_base = halfedges_.flat.extend(6,uninit);
  const FaceId f0 = f,
               f1(f_base),
               f2(f_base+1);
  const HalfedgeId en0(e_base+0), ep1(e_base+1),
                   en1(e_base+2), ep2(e_base+3),
                   en2(e_base+4), ep0(e_base+5);
  unsafe_link_face(f0,vec(e.x,en0,ep0));
  unsafe_link_face(f1,vec(e.y,en1,ep1));
  unsafe_link_face(f2,vec(e.z,en2,ep2));
  halfedges_[en0].src = v.y;
  halfedges_[en1].src = v.z;
  halfedges_[en2].src = v.x;
  halfedges_[ep0].src = halfedges_[ep1].src = halfedges_[ep2].src = c;
  vertex_to_edge_[c] = ep0;
}

VertexId HalfedgeMesh::split_face(FaceId f) {
  const auto c = add_vertex();
  split_face(f,c);
  return c;
}

bool HalfedgeMesh::is_flip_safe(HalfedgeId e0) const {
  if (!valid(e0) || is_boundary(e0))
    return false;
  const auto e1 = reverse(e0);
  if (is_boundary(e1))
    return false;
  const auto o0 = src(prev(e0)),
             o1 = src(prev(e1));
  return o0!=o1 && !halfedge(o0,o1).valid();
}

void HalfedgeMesh::flip_edge(HalfedgeId e) {
  if (!is_flip_safe(e))
    throw RuntimeError(format("HalfedgeMesh::flip_edge: edge flip %d is invalid",e.id));
  unsafe_flip_edge(e);
}

void HalfedgeMesh::unsafe_flip_edge(HalfedgeId e0) {
  const auto e1 = reverse(e0);
  const auto f0 = face(e0),
             f1 = face(e1);
  const auto n0 = next(e0), p0 = prev(e0),
             n1 = next(e1), p1 = prev(e1);
  const auto v0 = src(e0), o0 = src(p0),
             v1 = src(e1), o1 = src(p1);
  unsafe_link_face(f0,vec(e0,p1,n0));
  unsafe_link_face(f1,vec(e1,p0,n1));
  halfedges_[e0].src = o0;
  halfedges_[e1].src = o1;
  if (vertex_to_edge_[v0]==e0) vertex_to_edge_[v0] = n1;
  if (vertex_to_edge_[v1]==e1) vertex_to_edge_[v1] = n0;
}

void HalfedgeMesh::permute_vertices(RawArray<const int> permutation, bool check) {
  GEODE_ASSERT(n_vertices()==permutation.size());
  GEODE_ASSERT(n_vertices()==vertex_to_edge_.size()); // Require no erased vertices

  // Permute vertex_to_edge_ out of place
  Array<HalfedgeId> new_vertex_to_edge(vertex_to_edge_.size(),uninit);
  if (check) {
    new_vertex_to_edge.fill(HalfedgeId(erased_id));
    for (const auto v : vertices()) {
      const int pv = permutation[v.id];
      GEODE_ASSERT(new_vertex_to_edge.valid(pv));
      new_vertex_to_edge[pv] = vertex_to_edge_[v];
    }
    GEODE_ASSERT(!new_vertex_to_edge.contains(HalfedgeId(erased_id)));
  } else
    for (const auto v : vertices())
      new_vertex_to_edge[permutation[v.id]] = vertex_to_edge_[v];
  vertex_to_edge_.flat = new_vertex_to_edge;

  // The other arrays can be modified in place
  for (auto& e : halfedges_.flat)
    e.src = VertexId(permutation[e.src.id]);
}

void HalfedgeMesh::assert_consistent() const {
  // Check that all indices are in their valid ranges, that bidirectional links match, and a few other properties.
  GEODE_ASSERT(!(n_halfedges()&1));
  for (const auto v : vertices()) {
    const auto e = halfedge(v);
    if (e.valid())
      GEODE_ASSERT(valid(e) && src(e)==v);
  }
  for (const auto f : faces()) {
    const auto e = halfedge(f);
    GEODE_ASSERT(valid(e) && face(e)==f);
  }
  for (const auto e : halfedges()) {
    const auto f = face(e);
    const auto p = prev(e), n = next(e), r = reverse(e);
    GEODE_ASSERT(src(r)==src(n));
    GEODE_ASSERT(src(e)!=dst(e));
    GEODE_ASSERT(valid(p) && next(p)==e && face(p)==f);
    GEODE_ASSERT(valid(n) && prev(n)==e && face(n)==f);
    if (f.valid())
      GEODE_ASSERT(valid(f) && halfedges(f).contains(e));
    else
      GEODE_ASSERT(face(r).valid());
    const auto ce = common_halfedge(f,face(r));
    GEODE_ASSERT(ce.valid() && face(ce)==f && face(reverse(ce))==face(r));
  }

  // Check that all faces are triangles
  for (const auto f : faces()) {
    const auto e = halfedge(f);
    GEODE_ASSERT(e==next(next(next(e))));
  }

  // Check that no two halfedges share the same vertices
  {
    Hashtable<Vector<VertexId,2>> pairs;
    for (const auto e : halfedges())
      GEODE_ASSERT(pairs.set(vertices(e)));
  }

  // Check that all halfedges are reachable by swinging around their source vertices, and that
  // boundary vertices point to boundary halfedges.
  Field<bool,HalfedgeId> seen(n_halfedges());
  for (const auto v : vertices())
    if (!isolated(v)) {
      bool boundary = false;
      for (const auto e : outgoing(v)) {
        GEODE_ASSERT(src(e)==v);
        seen[e] = true;
        boundary |= is_boundary(e);
      }
      GEODE_ASSERT(boundary==is_boundary(v));
    }
  GEODE_ASSERT(!seen.flat.contains(false));
}

Array<Vector<int,3>> HalfedgeMesh::elements() const {
  Array<Vector<int,3>> tris(n_faces(),uninit);
  for (const auto f : faces())
    tris[f.idx()] = Vector<int,3>(vertices(f));
  return tris;
}

bool HalfedgeMesh::has_boundary() const {
  for (const auto v : vertices())
    if (is_boundary(v))
      return true;
  return false;
}

bool HalfedgeMesh::is_manifold() const {
  return !has_boundary();
}

bool HalfedgeMesh::is_manifold_with_boundary() const {
  for (const auto v : vertices()) {
    const auto start = halfedge(v);
    if (is_boundary(start)) { // If the first halfedge is a boundary, we need to check for a second
      auto e = start;
      for (;;) {
        e = left(e);
        if (e==start)
          break;
        if (is_boundary(e)) // If there are two boundary halfedges, this vertex is bad
          return false;
      }
    }
  }
  return true;
}

int HalfedgeMesh::degree(VertexId v) const {
  int degree = 0;
  for (GEODE_UNUSED auto _ : outgoing(v))
    degree++;
  return degree;
}

Nested<HalfedgeId> HalfedgeMesh::boundary_loops() const {
  Nested<HalfedgeId> loops;
  boost::dynamic_bitset<> seen(n_halfedges());
  for (const auto start : halfedges())
    if (is_boundary(start) && !seen[start.idx()]) {
      auto e = start;
      do {
        loops.flat.append(e);
        seen[e.idx()] = true;
        e = next(e);
      } while (e!=start);
      loops.offsets.const_cast_().append(loops.flat.size());
    }
  return loops;
}

void HalfedgeMesh::unsafe_erase_last_vertex() {
  const VertexId v(n_vertices()-1);
  // erase all incident faces
  while (!isolated(v))
    unsafe_erase_face(face(reverse(halfedge(v))));
  // Remove the vertex
  vertex_to_edge_.flat.pop();
}

void HalfedgeMesh::unsafe_erase_face(const FaceId f) {
  // Break halfedges away from this face
  const auto es = halfedges(f);
  for (const auto e : es) {
    halfedges_[e].face = FaceId();
    vertex_to_edge_[src(e)] = e; // Make sure we know the vertex is now a boundary
  }

  // Remove unconnected halfedges (higher first to not break as we erase)
  for (const auto e0 : es.sorted().reversed()) {
    const auto e1 = reverse(e0);
    if (is_boundary(e1)) {
      // Decouple the halfedge pair from its neighbors
      const auto v0 = src(e0), v1 = src(e1);
      const auto p0 = prev(e0), n0 = next(e0),
                 p1 = prev(e1), n1 = next(e1);
      if (halfedge(v0)==e0)
        vertex_to_edge_[v0] = n1!=e0?n1:HalfedgeId();
      if (halfedge(v1)==e1)
        vertex_to_edge_[v1] = n0!=e1?n0:HalfedgeId();

      GEODE_ASSERT(halfedge(v0).id<n_halfedges());;
      GEODE_ASSERT(halfedge(v1).id<n_halfedges());;

      unsafe_link(p0,n1);
      unsafe_link(p1,n0);
      // Rename the last halfedge pair to e0,e1
      const int new_count = n_halfedges()-2;
      if (e0.id<new_count) {
        const HalfedgeId e2(new_count+0),
                         e3(new_count+1);
        const auto v2 = src(e2), v3 = src(e3);
        const auto p2 = prev(e2), n2 = next(e2),
                   p3 = prev(e3), n3 = next(e3);
        const auto f2 = face(e2), f3 = face(e3);
        halfedges_[e0] = halfedges_[e2];
        halfedges_[e1] = halfedges_[e3];
        unsafe_link(p2,e0);
        unsafe_link(e0,n2);
        unsafe_link(p3,e1);
        unsafe_link(e1,n3);
        if (f2.valid())
          face_to_edge_[f2] = e0;
        if (f3.valid())
          face_to_edge_[f3] = e1;
        if (halfedge(v2)==e2)
          vertex_to_edge_[v2] = e0;
        if (halfedge(v3)==e3)
          vertex_to_edge_[v3] = e1;
      }
      // Discard the erased halfedges
      halfedges_.flat.pop();
      halfedges_.flat.pop();

      GEODE_ASSERT(halfedge(v0).id<n_halfedges());;
      GEODE_ASSERT(halfedge(v1).id<n_halfedges());;
    }
  }

  // Rename the last face to f
  if (f.id<n_faces()-1) {
    const auto es = halfedges(FaceId(n_faces()-1));
    halfedges_[es.x].face = halfedges_[es.y].face = halfedges_[es.z].face = f;
    face_to_edge_[f] = es.x;
  }

  // Remove the erase face
  face_to_edge_.flat.pop();
}

void HalfedgeMesh::dump_internals() const {
  cout << format("halfedge dump:\n  vertices: %d\n",n_vertices());
  for (const auto v : vertices()) {
    if (isolated(v))
      cout << format("    v%d: e_\n",v.id);
    else {
      // Print all outgoing halfedges, taking care not to crash or loop forever if the structure is invalid
      cout << format("    v%d:",v.id);
      const auto start = halfedge(v);
      auto e = start;
      int limit = n_edges()+1;
      do {
        cout << " e"<<e.id;
        if (!valid(e) || !valid(prev(e))) {
          cout << " boom";
          break;
        } else if (!--limit) {
          cout << " ...";
          break;
        } else
          e = reverse(prev(e));
      } while (e!=start);
      cout << (valid(start)?is_boundary(start)?", b\n":", i\n":"\n");
    }
  }
  cout << format("  halfedges: %d\n",n_halfedges());
  for (const auto e : halfedges())
    cout << format("    e%d: v%d v%d, e%d e%d, f%s\n",e.id,src(e).id,dst(e).id,prev(e).id,next(e).id,is_boundary(e)?"_":str(face(e)).c_str());
  cout << format("  faces: %d\n",n_faces());
  for (const auto f : faces()) {
    const auto e = halfedges(f);
    cout << format("    f%d: e%d e%d e%d\n",f.id,e.x.id,e.y.id,e.z.id);
  }
  cout << endl;
}

static int random_edge_flips(HalfedgeMesh& mesh, const int attempts, const uint128_t key) {
  int flips = 0;
  if (mesh.n_halfedges()) {
    const auto random = new_<Random>(key);
    for (int a=0;a<attempts;a++) {
      const HalfedgeId e(random->uniform<int>(0,mesh.n_halfedges()));
      if (mesh.is_flip_safe(e)) {
        mesh.unsafe_flip_edge(e);
        flips++;
      }
    }
  }
  return flips;
}

static void random_face_splits(HalfedgeMesh& mesh, const int splits, const uint128_t key) {
  if (mesh.n_faces()) {
    const auto random = new_<Random>(key);
    for (int a=0;a<splits;a++) {
      const FaceId f(random->uniform<int>(0,mesh.n_faces()));
      const auto v = mesh.split_face(f);
      GEODE_ASSERT(mesh.face(mesh.halfedge(v))==f);
    }
  }
}

static void mesh_destruction_test(HalfedgeMesh& mesh, const uint128_t key) {
  const auto random = new_<Random>(key);
  while (mesh.n_vertices()) {
    const int target = random->uniform<int>(0,1+2*mesh.n_faces());
    if (target<mesh.n_faces())
      mesh.unsafe_erase_face(FaceId(target));
    else
      mesh.unsafe_erase_last_vertex();
    mesh.assert_consistent();
  }
}

}
using namespace geode;

void wrap_halfedge_mesh() {
  typedef HalfedgeMesh Self;
  Class<Self>("HalfedgeMesh")
    .GEODE_INIT()
    .GEODE_METHOD(copy)
    .GEODE_GET(n_vertices)
    .GEODE_GET(n_halfedges)
    .GEODE_GET(n_edges)
    .GEODE_GET(n_faces)
    .GEODE_GET(chi)
    .GEODE_METHOD(elements)
    .GEODE_METHOD(has_boundary)
    .GEODE_METHOD(is_manifold)
    .GEODE_METHOD(is_manifold_with_boundary)
    .GEODE_METHOD(boundary_loops)
    .GEODE_METHOD(add_vertex)
    .GEODE_METHOD(add_vertices)
    .GEODE_METHOD(add_face)
    .GEODE_METHOD(add_faces)
    .GEODE_METHOD(assert_consistent)
    .GEODE_METHOD(dump_internals)
    ;

  // For testing purposes
  GEODE_FUNCTION(random_edge_flips)
  GEODE_FUNCTION(random_face_splits)
  GEODE_FUNCTION(mesh_destruction_test)
}
