// A halfedge data structure representing oriented triangle meshes.

#include <other/core/mesh/HalfedgeMesh.h>
#include <other/core/array/convert.h>
#include <other/core/array/NestedArray.h>
#include <other/core/python/Class.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/Log.h>
#include <other/core/vector/convert.h>
#include <boost/dynamic_bitset.hpp>
namespace other {

using Log::cout;
using std::endl;

OTHER_DEFINE_TYPE(HalfedgeMesh)

// Add numpy conversion support
#ifdef OTHER_PYTHON
namespace {
template<> struct NumpyIsScalar<VertexId>:public mpl::true_{};
template<> struct NumpyIsScalar<HalfedgeId>:public mpl::true_{};
template<> struct NumpyIsScalar<FaceId>:public mpl::true_{};
template<> struct NumpyScalar<VertexId>{enum{value=NPY_INT};};
template<> struct NumpyScalar<HalfedgeId>{enum{value=NPY_INT};};
template<> struct NumpyScalar<FaceId>{enum{value=NPY_INT};};
}
OTHER_DEFINE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,3,VertexId)
#endif

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

OTHER_COLD static void add_face_error(const Vector<VertexId,3> v, const char* reason) {
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
  // together into components.  This arbitrary choice may be incompatible with the new face, so a bit of surgery may
  // be required to correct the linkage.  We check for all errors before we do any actual surgery so that add_face
  // makes changes only when it succeeds.
  #define PREPARE(a,b,c) \
    const auto c = a.valid() && b.valid() && next(a)!=b ? right_around_dst_to_boundary(*this,b) : HalfedgeId(); \
    if (c.valid() && b==c) \
      add_face_error(v,", result would consist of a closed ring plus extra triangles");
  PREPARE(e0,e1,c0)
  PREPARE(e1,e2,c1)
  PREPARE(e2,e0,c2)
  // For each pair of adjacent ei,ej, we want next(ei)==ej.  We've already computed ci = the other side of ej's triangle fan.
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
    int e_next = halfedges_.size();
    halfedges_.flat.resize(e_next+new_edges,false);
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

VertexId HalfedgeMesh::split_face(FaceId f) {
  const auto e = halfedges(f);
  const auto v = vertices(f);
  const auto c = add_vertex();
  const int f_base = face_to_edge_.size();
  const int e_base = halfedges_.size();
  face_to_edge_.flat.resize(f_base+2,false);
  halfedges_.flat.resize(e_base+6,false);
  const FaceId f0 = f,
               f1(f_base),
               f2(f_base+1);
  const HalfedgeId en0(e_base+0), ep1(e_base+1),
                   en1(e_base+2), ep2(e_base+3),
                   en2(e_base+4), ep0(e_base+5);
  unsafe_link_face(f0,vec(e.x,en0,ep0));
  unsafe_link_face(f1,vec(e.y,en1,ep1));
  unsafe_link_face(f2,vec(e.z,en2,ep2));
  halfedges_[en0].src = v.x;
  halfedges_[en1].src = v.y;
  halfedges_[en2].src = v.z;
  halfedges_[ep0].src = halfedges_[ep1].src = halfedges_[ep2].src = c;
  vertex_to_edge_[c] = ep0;
  return c;
}

bool HalfedgeMesh::is_flip_safe(HalfedgeId e0) const {
  if (!valid(e0) || !is_boundary(e0))
    return false;
  const auto e1 = reverse(e0);
  if (!is_boundary(e1))
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

void HalfedgeMesh::assert_consistent() const {
  // Check that all indices are in their valid ranges, that bidirectional links match, and a few other properties.
  OTHER_ASSERT(!(n_halfedges()&1));
  for (const auto v : vertices()) {
    const auto e = halfedge(v);
    if (e.valid())
      OTHER_ASSERT(valid(e) && src(e)==v);
  }
  for (const auto f : faces()) {
    const auto e = halfedge(f);
    OTHER_ASSERT(valid(e) && face(e)==f);
  }
  for (const auto e : halfedges()) {
    const auto f = face(e);
    const auto p = prev(e), n = next(e), r = reverse(e);
    OTHER_ASSERT(src(r)==src(n));
    OTHER_ASSERT(src(e)!=dst(e));
    OTHER_ASSERT(valid(p) && next(p)==e && face(p)==f);
    OTHER_ASSERT(valid(n) && prev(n)==e && face(n)==f);
    if (f.valid())
      OTHER_ASSERT(valid(f) && halfedges(f).contains(e));
    else
      OTHER_ASSERT(face(r).valid());
  }

  // Check that all faces are triangles
  for (const auto f : faces()) {
    const auto e = halfedge(f);
    OTHER_ASSERT(e==next(next(next(e))));
  }

  // Check that no two halfedges share the same vertices
  {
    Hashtable<Vector<VertexId,2>> pairs;
    for (const auto e : halfedges())
      OTHER_ASSERT(pairs.set(vertices(e)));
  }

  // Check that all halfedges are reachable by swinging around their source vertices, and that
  // boundary vertices point to boundary halfedges.
  Field<bool,HalfedgeId> seen(n_halfedges());
  for (const auto v : vertices())
    if (!isolated(v)) {
      bool boundary = false;
      for (const auto e : outgoing(v)) {
        OTHER_ASSERT(src(e)==v);
        seen[e] = true;
        boundary |= is_boundary(e);
      }
      OTHER_ASSERT(boundary==is_boundary(v));
    }
  OTHER_ASSERT(!seen.flat.contains(false));
}

Array<Vector<int,3>> HalfedgeMesh::elements() const {
  Array<Vector<int,3>> tris(n_faces(),false);
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

NestedArray<HalfedgeId> HalfedgeMesh::boundary_loops() const {
  NestedArray<HalfedgeId> loops;
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

static void random_edge_flips(HalfedgeMesh& mesh, const int attempts, const uint128_t key) {
  const auto random = new_<Random>(key);
  for (int a=0;a<attempts;a++) {
    const HalfedgeId e(random->uniform<int>(0,mesh.n_halfedges()));
    if (mesh.is_flip_safe(e))
      mesh.unsafe_flip_edge(e);
  }
}

}
using namespace other;

void wrap_halfedge_mesh() {
  typedef HalfedgeMesh Self;
  Class<Self>("HalfedgeMesh")
    .OTHER_INIT()
    .OTHER_METHOD(copy)
    .OTHER_GET(n_vertices)
    .OTHER_GET(n_halfedges)
    .OTHER_GET(n_edges)
    .OTHER_GET(n_faces)
    .OTHER_GET(chi)
    .OTHER_METHOD(elements)
    .OTHER_METHOD(has_boundary)
    .OTHER_METHOD(is_manifold)
    .OTHER_METHOD(is_manifold_with_boundary)
    .OTHER_METHOD(boundary_loops)
    .OTHER_METHOD(add_vertex)
    .OTHER_METHOD(add_vertices)
    .OTHER_METHOD(add_face)
    .OTHER_METHOD(add_faces)
    .OTHER_METHOD(assert_consistent)
    .OTHER_METHOD(dump_internals)
    ;

  // For testing purposes
  OTHER_FUNCTION(random_edge_flips)
}
