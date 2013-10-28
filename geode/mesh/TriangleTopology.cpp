// A corner data structure representing oriented triangle meshes.

#include <geode/mesh/TriangleTopology.h>
#include <geode/array/convert.h>
#include <geode/array/Nested.h>
#include <geode/python/Class.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/Log.h>
#include <geode/vector/convert.h>
#include <boost/dynamic_bitset.hpp>
namespace geode {

using Log::cout;
using std::endl;

GEODE_DEFINE_TYPE(TriangleTopology)

// Add numpy conversion support
#ifdef GEODE_PYTHON
namespace {
template<> struct NumpyIsScalar<VertexId>:public mpl::true_{};
template<> struct NumpyIsScalar<HalfedgeId>:public mpl::true_{};
template<> struct NumpyIsScalar<FaceId>:public mpl::true_{};
template<> struct NumpyScalar<VertexId>{enum{value=NPY_INT};};
template<> struct NumpyScalar<HalfedgeId>{enum{value=NPY_INT};};
template<> struct NumpyScalar<FaceId>{enum{value=NPY_INT};};
}
#endif

static string str_halfedge(HalfedgeId e) {
  return e.valid() ? e.id>=0 ? format("e%d%d",e.id/3,e.id%3)
                             : format("b%d",-1-e.id)
                   : "e_";
}

TriangleTopology::TriangleTopology()
  : n_vertices_(0)
  , n_faces_(0)
  , n_boundary_edges_(0) {}

TriangleTopology::TriangleTopology(const TriangleTopology& mesh)
  : n_vertices_(mesh.n_vertices_)
  , n_faces_(mesh.n_faces_)
  , n_boundary_edges_(mesh.n_boundary_edges_)
  , faces_(mesh.faces_.copy())
  , vertex_to_edge_(mesh.vertex_to_edge_.copy())
  , boundaries_(mesh.boundaries_.copy())
  , erased_boundaries_(mesh.erased_boundaries_) {}

TriangleTopology::TriangleTopology(RawArray<const Vector<int,3>> faces)
: TriangleTopology() {
  add_faces(faces);
}

TriangleTopology::TriangleTopology(TriangleSoup const &soup)
: TriangleTopology(RawArray<const Vector<int,3>>(soup.elements)) {
}

TriangleTopology::~TriangleTopology() {}

Ref<TriangleTopology> TriangleTopology::copy() const {
  return new_<TriangleTopology>(*this);
}

HalfedgeId TriangleTopology::halfedge(VertexId v0, VertexId v1) const {
  assert(valid(v0));
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

HalfedgeId TriangleTopology::common_halfedge(FaceId f0, FaceId f1) const {
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

inline static HalfedgeId apply_halfedge_permutation(HalfedgeId h, RawArray<const int> face_permutation, RawArray<const int> boundary_permutation) {
  assert(h.id != erased_id);
  if (!h.valid())
    return h;
  if (h.id >= 0)
    return HalfedgeId(3*face_permutation[h.id/3] + h.id%3);
  else
    return HalfedgeId(-1-boundary_permutation[-1-h.id]);
}

Tuple<Array<int>, Array<int>, Array<int>> TriangleTopology::add(TriangleTopology const &other) {
  Array<int> vertex_permutation(other.vertex_to_edge_.size()),
             face_permutation(other.faces_.size()),
             boundary_permutation(boundaries_.size());

  // the first indices
  int base_vertex = vertex_to_edge_.size();
  int base_face = faces_.size();
  int base_boundary = boundaries_.size();

  // add things from other to the end
  vertex_permutation.fill(-1);
  for (int i = 0; i < vertices_size_)
  for (auto vi : other.vertices()) {
    vertex_permutation[vi.i.id] = vertex_to_edge_.size();
    vertex_to_edge_.append(other.vertex_to_edge_[vi.i]);
  }
  face_permutation.fill(-1);
  for (auto fi : other.faces()) {
    face_permutation[fi.i.id] = faces_.size();
    faces_.append(other.faces_[fi.i]);
  }
  boundary_permutation.fill(-1);
  for (auto bi : other.boundary_edges()) {
    boundary_permutation[bi.i.id] = boundaries_.size();
    boundaries_.append(other.boundaries_[bi.i]);
  }

  // renumber all new primitives

  // reflect id changes in newly added arrays
  for (auto& h : vertex_to_edge_.flat.slice(base_vertex, vertex_to_edge_.size())) {
    assert(h.id != erased_id);
    h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& f : faces_.flat.slice(base_face, faces_.size())) {
    assert(f.vertices.x.id != erased_id);
    for (auto& v : f.vertices)
      v = VertexId(vertex_permutation[v.id]);
    for (auto &h : f.neighbors)
      h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& b : boundaries_.slice(base_boundary, boundaries_.size())) {
    assert(b.src.id != erased_id);
    assert(b.src.valid());
    b.src = VertexId(vertex_permutation[b.src.id]);
    b.next = apply_halfedge_permutation(b.next, face_permutation, boundary_permutation);
    b.prev = apply_halfedge_permutation(b.prev, face_permutation, boundary_permutation);
    b.reverse = apply_halfedge_permutation(b.reverse, face_permutation, boundary_permutation);
  }

  return tuple(vertex_permutation, face_permutation, boundary_permutation);
}

VertexId TriangleTopology::add_vertex() {
  n_vertices_++;
  return vertex_to_edge_.append(HalfedgeId());
}

VertexId TriangleTopology::add_vertices(int n) {
  int id = n_vertices_;
  n_vertices_ += n;
  vertex_to_edge_.flat.resize(vertex_to_edge_.size()+n);
  return VertexId(id);
}

static inline HalfedgeId right_around_dst_to_boundary(const TriangleTopology& mesh, HalfedgeId e) {
  e = mesh.reverse(mesh.next(mesh.reverse(e)));
  while (!mesh.is_boundary(e))
    e = mesh.reverse(mesh.next(e));
  return e;
}

// Set halfedge(v) to an outgoing boundary halfedge if possible, or any halfedge otherwise.  We start
// looping at the given initial guess, which should ideally be a likely boundary edge.
static inline void fix_vertex_to_edge(TriangleTopology& mesh, const VertexId v, const HalfedgeId start) {
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

// Allocate a fresh boundary edge and set its src and reverse pointers
static inline HalfedgeId unsafe_new_boundary(TriangleTopology& mesh, const VertexId src, const HalfedgeId reverse) {
  mesh.n_boundary_edges_++;
  HalfedgeId e;
  if (mesh.erased_boundaries_.valid()) {
    e = mesh.erased_boundaries_;
    mesh.erased_boundaries_ = mesh.boundaries_[-1-e.id].next;
  } else {
    const int b = mesh.boundaries_.size();
    mesh.boundaries_.resize(b+1,false);
    e = HalfedgeId(-1-b);
  }
  mesh.boundaries_[-1-e.id].src = src;
  mesh.boundaries_[-1-e.id].reverse = reverse;
  return e;
}

GEODE_COLD static void add_face_error(const Vector<VertexId,3> v, const char* reason) {
  throw RuntimeError(format("TriangleTopology::add_face: can't add face (%d,%d,%d)%s",v.x.id,v.y.id,v.z.id,reason));
}

FaceId TriangleTopology::add_face(const Vector<VertexId,3> v) {
  // Check for errors
  if (!valid(v.x) || !valid(v.y) || !valid(v.z))
    add_face_error(v," containing invalid vertex");
  if (v.x==v.y || v.y==v.z || v.z==v.x)
    add_face_error(v," containing duplicate vertex");
  if (!is_boundary(v.x) || !is_boundary(v.y) || !is_boundary(v.z))
    add_face_error(v,", one of the vertices is interior");
  const auto e0 = halfedge(v.x,v.y),
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
      unsafe_boundary_link(a,b); \
      unsafe_boundary_link(c,na); \
      unsafe_boundary_link(pb,nc); \
    }
  RELINK(e0,e1,c0)
  RELINK(e1,e2,c1)
  RELINK(e2,e0,c2)

  // Create a new face, including three implicit halfedges.
  n_faces_++;
  const auto f = faces_.append(FaceInfo());
  faces_[f].vertices = v;

  // Look up all connectivity we'll need to restructure the mesh.  This includes the reverses
  // r0,r1,r2 of each of e0,e1,e2 (the old halfedges of the triangle), and their future prev/next links.
  // If r0,r1,r2 don't exist yet, they are allocated.
  const auto ve0 = halfedge(v.x),
             ve1 = halfedge(v.y),
             ve2 = halfedge(v.z);
  #define REVERSE(i) \
    const auto r##i = e##i.valid() ? boundaries_[-1-e##i.id].reverse \
                                   : unsafe_new_boundary(*this,v[(i+1)%3],HalfedgeId(3*f.id+i));
  REVERSE(0)
  REVERSE(1)
  REVERSE(2)
  #define NEAR(i,in,ip) \
    HalfedgeId prev##i, next##i; \
    if (!e##i.valid()) { \
      prev##i = ve##in.valid() ? boundaries_[-1-(e##in.valid() ? e##in : ve##in).id].prev : r##in; \
      next##i = ve##i.valid() ? e##ip.valid() ? boundaries_[-1-e##ip.id].next : ve##i : r##ip; \
    }
  NEAR(0,1,2)
  NEAR(1,2,0)
  NEAR(2,0,1)

  // Link everything together
  #define LINK(i) \
    unsafe_set_reverse(f,i,r##i); \
    if (!e##i.valid()) { \
      unsafe_boundary_link(prev##i,r##i); \
      unsafe_boundary_link(r##i,next##i); \
    }
  LINK(0)
  LINK(1)
  LINK(2)

  // erase edges that used to be boundaries.  We do this after allocating new boundary edges
  // so that fields we need are not overwritten before they are used.
  if (e0.valid()) unsafe_set_erased(e0);
  if (e1.valid()) unsafe_set_erased(e1);
  if (e2.valid()) unsafe_set_erased(e2);

  // Fix vertex to edge pointers to point to boundary halfedges if possible
  fix_vertex_to_edge(*this,v.x,r2);
  fix_vertex_to_edge(*this,v.y,r0);
  fix_vertex_to_edge(*this,v.z,r1);

  // All done!
  return f;
}

FaceId TriangleTopology::add_faces(RawArray<const Vector<int,3>> vs) {
  // TODO: We desperately need a batch insertion routine.
  if (vs.empty()) {
    return FaceId();
  } else {
    FaceId first(faces_.size());
    for (auto& v : vs)
      add_face(Vector<VertexId,3>(v));
    return first;
  }
}

void TriangleTopology::split_face(const FaceId f, const VertexId c) {
  GEODE_ASSERT(valid(f) && isolated(c));
  const auto v = faces_[f].vertices;
  const auto n = faces_[f].neighbors;
  const int f_base = faces_.size();
  n_faces_ += 2;
  faces_.flat.resize(f_base+2,false);
  const auto fs = vec(f,FaceId(f_base),FaceId(f_base+1));
  #define UPDATE(i) { \
    const int ip = (i+2)%3, in = (i+1)%3; \
    faces_[fs[i]].vertices.set(v[i],v[in],c); \
    unsafe_set_reverse(fs[i],0,n[i]); \
    faces_[fs[i]].neighbors[1] = HalfedgeId(3*fs[in].id+2); \
    faces_[fs[i]].neighbors[2] = HalfedgeId(3*fs[ip].id+1); \
    if (i && vertex_to_edge_[v[i]].id==3*f.id+i) \
      vertex_to_edge_[v[i]] = HalfedgeId(3*fs[i].id); }
  UPDATE(0)
  UPDATE(1)
  UPDATE(2)
  #undef UPDATE
  vertex_to_edge_[c] = halfedge(f,2);
}

VertexId TriangleTopology::split_face(FaceId f) {
  const auto c = add_vertex();
  split_face(f,c);
  return c;
}

bool TriangleTopology::is_flip_safe(HalfedgeId e0) const {
  if (!valid(e0) || is_boundary(e0))
    return false;
  const auto e1 = reverse(e0);
  if (is_boundary(e1))
    return false;
  const auto o0 = src(prev(e0)),
             o1 = src(prev(e1));
  return o0!=o1 && !halfedge(o0,o1).valid();
}

HalfedgeId TriangleTopology::flip_edge(HalfedgeId e) {
  if (!is_flip_safe(e))
    throw RuntimeError(format("TriangleTopology::flip_edge: edge flip %d is invalid",e.id));
  return unsafe_flip_edge(e);
}

HalfedgeId TriangleTopology::unsafe_flip_edge(HalfedgeId e0) {
  const auto e1 = reverse(e0);
  const auto f0 = face(e0),
             f1 = face(e1);
  const auto n0 = next(e0), p0 = prev(e0),
             n1 = next(e1), p1 = prev(e1),
             rn0 = reverse(n0), rp0 = reverse(p0),
             rn1 = reverse(n1), rp1 = reverse(p1);
  const auto v0 = src(e0), o0 = src(p0),
             v1 = src(e1), o1 = src(p1);
  faces_[f0].vertices = vec(o0,o1,v1);
  faces_[f1].vertices = vec(o1,o0,v0);
  faces_[f0].neighbors.x = HalfedgeId(3*f1.id);
  faces_[f1].neighbors.x = HalfedgeId(3*f0.id);
  unsafe_set_reverse(f0,1,rp1);
  unsafe_set_reverse(f0,2,rn0);
  unsafe_set_reverse(f1,1,rp0);
  unsafe_set_reverse(f1,2,rn1);
  // Fix vertex to edge links
  auto &ve0 = vertex_to_edge_[v0],
       &ve1 = vertex_to_edge_[v1],
       &oe0 = vertex_to_edge_[o0],
       &oe1 = vertex_to_edge_[o1];
  if (ve0==e0 || ve0==n1) ve0 = HalfedgeId(3*f1.id+2);
  if (ve1==e1 || ve1==n0) ve1 = HalfedgeId(3*f0.id+2);
  if (oe0==p0) oe0 = HalfedgeId(3*f0.id);
  if (oe1==p1) oe1 = HalfedgeId(3*f1.id);
  return HalfedgeId(3*f0.id);
}

void TriangleTopology::assert_consistent() const {
  // Check simple vertex properties
  int actual_vertices = 0;
  for (const auto v : vertices()) {
    actual_vertices++;
    const auto e = halfedge(v);
    if (e.valid())
      GEODE_ASSERT(src(e)==v);
  }
  GEODE_ASSERT(actual_vertices==n_vertices());

  // Check simple face properties
  int actual_faces = 0;
  for (const auto f : faces()) {
    actual_faces++;
    const auto es = halfedges(f);
    GEODE_ASSERT(es.x==halfedge(f));
    for (int i=0;i<3;i++) {
      const auto e = es[i];
      GEODE_ASSERT(valid(e) && face(e)==f);
      GEODE_ASSERT(src(e)==vertex(f,i) && dst(e)==vertex(f,(i+1)%3));
      GEODE_ASSERT(next(e)==halfedge(f,(i+1)%3));
    }
  }
  GEODE_ASSERT(actual_faces==n_faces());

  // Check simple edge properties
  GEODE_ASSERT(!((3*n_faces()+n_boundary_edges())&1));
  for (const auto e : halfedges()) {
    const auto f = face(e);
    const auto p = prev(e), n = next(e), r = reverse(e);
    GEODE_ASSERT(valid(p));
    GEODE_ASSERT(valid(n));
    GEODE_ASSERT(valid(r));
    GEODE_ASSERT(e!=p && e!=n && e!=r);
    GEODE_ASSERT(src(r)==src(n));
    GEODE_ASSERT(src(e)!=dst(e));
    GEODE_ASSERT(valid(p) && next(p)==e && face(p)==f);
    GEODE_ASSERT(valid(n) && prev(n)==e && face(n)==f);
    if (f.valid())
      GEODE_ASSERT(valid(f) && halfedges(f).contains(e));
    else {
      GEODE_ASSERT(is_boundary(e));
      GEODE_ASSERT(is_boundary(p));
      GEODE_ASSERT(is_boundary(n));
      GEODE_ASSERT(!is_boundary(r));
      GEODE_ASSERT(face(r).valid());
    }
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
  boost::dynamic_bitset<> seen(boundaries_.size()+3*faces_.size());
  for (const auto v : vertices())
    if (!isolated(v)) {
      bool boundary = false;
      for (const auto e : outgoing(v)) {
        GEODE_ASSERT(src(e)==v);
        seen[boundaries_.size()+e.id] = true;
        boundary |= is_boundary(e);
      }
      GEODE_ASSERT(boundary==is_boundary(v));
    }
  GEODE_ASSERT(seen.count()==size_t(2*n_edges()));

  // Check that all erased boundary edges occur in our linked list
  int limit = boundaries_.size();
  int actual_erased = 0;
  for (auto b=erased_boundaries_;b.valid();b=boundaries_[-1-b.id].next) {
    GEODE_ASSERT(boundaries_.valid(-1-b.id));
    GEODE_ASSERT(limit--);
    actual_erased++;
  }
  GEODE_ASSERT(n_boundary_edges()+actual_erased==boundaries_.size());
}

Array<Vector<int,3>> TriangleTopology::elements() const {
  Array<Vector<int,3>> tris(n_faces(),false);
  int i = 0;
  for (const auto f : faces())
    tris[i++] = Vector<int,3>(faces_[f].vertices);
  return tris;
}

bool TriangleTopology::has_boundary() const {
  return n_boundary_edges()!=0;
}

bool TriangleTopology::is_manifold() const {
  return !has_boundary();
}

bool TriangleTopology::is_manifold_with_boundary() const {
  if (is_manifold()) // Finish in O(1) time if possible
    return true;
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

bool TriangleTopology::has_isolated_vertices() const {
  for (const auto v : vertices())
    if (isolated(v))
      return true;
  return false;
}

int TriangleTopology::degree(VertexId v) const {
  int degree = 0;
  for (GEODE_UNUSED auto _ : outgoing(v))
    degree++;
  return degree;
}

Nested<HalfedgeId> TriangleTopology::boundary_loops() const {
  Nested<HalfedgeId> loops;
  boost::dynamic_bitset<> seen(boundaries_.size());
  for (const auto start : boundary_edges())
    if (!seen[-1-start.id]) {
      auto e = start;
      do {
        loops.flat.append(e);
        seen[-1-e.id] = true;
        e = next(e);
      } while (e!=start);
      loops.offsets.const_cast_().append(loops.flat.size());
    }
  return loops;
}

void TriangleTopology::erase_last_vertex_with_reordering() {
  const VertexId v(vertex_to_edge_.size()-1);
  // erase all incident faces
  while (!isolated(v))
    erase_face_with_reordering(face(reverse(halfedge(v))));
  // Remove the vertex
  vertex_to_edge_.flat.pop();
  n_vertices_--;
}

void TriangleTopology::erase_face_with_reordering(const FaceId f) {
  // Look up connectivity of neighboring boundary edges, then erase them
  const auto e = faces_[f].neighbors;
  Vector<HalfedgeId,2> near[3]; // (prev,next) for each neighbor boundary edge
  for (int i=0;i<3;i++)
    if (e[i].id<0) {
      const auto& B = boundaries_[-1-e[i].id];
      near[i] = vec(B.prev,B.next);
      unsafe_set_erased(e[i]);
    }

  // Create any new boundary edges and set src and reverse
  HalfedgeId b[3];
  for (int i=0;i<3;i++)
    if (e[i].id>=0) {
      b[i] = unsafe_new_boundary(*this,faces_[f].vertices[i],e[i]);
      const int fi = e[i].id/3;
      faces_.flat[fi].neighbors[e[i].id-3*fi] = b[i];
    }

  // Fix connectivity around each vertex: link together prev/next adjacent edges and update vertex to edge pointers
  for (int i=0;i<3;i++) {
    const int ip = i?i-1:2;
    const auto v = faces_[f].vertices[i];
    const auto prev = e[ip].id>=0 ? b[ip] : near[ip].x,
               next = e[i ].id>=0 ? b[i ] : near[i ].y;
    if (e[i].id>=0 || e[ip].id>=0 || prev!=e[i]) {
      unsafe_boundary_link(prev,next);
      vertex_to_edge_[v] = next;
    } else if (vertex_to_edge_[v]==e[ip])
      vertex_to_edge_[v] = HalfedgeId();
  }

  // Rename the last face to f
  const FaceId f1(faces_.size()-1);
  if (f.id<f1.id) {
    assert(!erased(f1));
    const auto I = faces_[f1];
    faces_[f].vertices = I.vertices;
    for (int i=0;i<3;i++) {
      unsafe_set_reverse(f,i,I.neighbors[i]);
      if (vertex_to_edge_[I.vertices[i]].id==3*f1.id+i)
        vertex_to_edge_[I.vertices[i]].id = 3*f.id+i;
    }
  }

  // Remove the erased face
  faces_.flat.pop();
  n_faces_--;
}

void TriangleTopology::dump_internals() const {
  cout << format("corner mesh dump:\n  vertices: %d\n",n_vertices());
  for (const auto v : vertices()) {
    if (isolated(v))
      cout << format("    v%d: e_\n",v.id);
    else {
      // Print all outgoing halfedges, taking care not to crash or loop forever if the structure is invalid
      cout << format("    v%d:",v.id);
      const auto start = halfedge(v);
      auto e = start;
      int limit = n_faces()+1;
      do {
        cout << ' '<<str_halfedge(e);
        if (!valid(e) || !valid(prev(e))) {
          cout << " boom";
          break;
        } else if (!limit--) {
          cout << " ...";
          break;
        } else
          e = reverse(prev(e));
      } while (e!=start);
      cout << (valid(start)?is_boundary(start)?", b\n":", i\n":"\n");
    }
  }
  cout << format("  faces: %d\n",n_faces());
  for (const auto f : faces()) {
    const auto v = vertices(f);
    const auto n = faces_[f].neighbors;
    cout << format("    f%d: v%d v%d v%d, %s %s %s\n",f.id,v.x.id,v.y.id,v.z.id,str_halfedge(n.x),str_halfedge(n.y),str_halfedge(n.z));
  }
  cout << format("  boundary edges: %d\n",n_boundary_edges());
  for (const auto e : boundary_edges()) {
    const auto n = next(e);
    cout << format("    %s: v%d %s, %s %s, r %s\n",str_halfedge(e),src(e).id,(valid(n)?format("v%d",src(n).id):"boom"),str_halfedge(prev(e)),str_halfedge(n),str_halfedge(reverse(e)));
  }
  cout << endl;
}

void TriangleTopology::permute_vertices(RawArray<const int> permutation, bool check) {
  GEODE_ASSERT(n_vertices()==permutation.size());
  GEODE_ASSERT(n_vertices()==vertex_to_edge_.size()); // Require no erased vertices

  // Permute vertex_to_edge_ out of place
  Array<HalfedgeId> new_vertex_to_edge(vertex_to_edge_.size(),false);
  if (check) {
    new_vertex_to_edge.fill(HalfedgeId(erased_id));
    for (const auto v : all_vertices()) {
      const int pv = permutation[v.id];
      GEODE_ASSERT(new_vertex_to_edge.valid(pv));
      new_vertex_to_edge[pv] = vertex_to_edge_[v];
    }
    GEODE_ASSERT(!new_vertex_to_edge.contains(HalfedgeId(erased_id)));
  } else
    for (const auto v : all_vertices())
      new_vertex_to_edge[permutation[v.id]] = vertex_to_edge_[v];
  vertex_to_edge_.flat = new_vertex_to_edge;

  // The other arrays can be modified in place
  for (auto& f : faces_.flat)
    if (f.vertices.x.id!=erased_id)
      for (auto& v : f.vertices)
        v = VertexId(permutation[v.id]);
  for (auto& b : boundaries_)
    if (b.src.id!=erased_id)
      b.src = VertexId(permutation[b.src.id]);
}

// erase the given vertex. erases all incident faces. If erase_isolated is true, also erase other vertices that are now isolated.
void TriangleTopology::erase(VertexId id, bool erase_isolated) {
  // TODO: Make a better version of this. For now, just erase all incident faces
  // and then our own (or have it automatically erased if erase_isolated is true)
  GEODE_ASSERT(!erased(id));

  while (!isolated(id))
    erase(face(reverse(halfedge(id))));

  // erase at least this vertex, if not already happened
  if (!erase_isolated) {
    unsafe_set_erased(id);
  }
}

// erase the given halfedge. erases all incident faces as well. If erase_isolated is true, also erase incident vertices that are now isolated.
void TriangleTopology::erase(HalfedgeId id, bool erase_isolated) {
  auto he = faces(id);
  for (auto h : he) {
    if (valid(h)) {
      erase(h, erase_isolated);
    }
  }
}

// erase the given face. If erase_isolated is true, also erases incident vertices that are now isolated.
void TriangleTopology::erase(FaceId f, bool erase_isolated) {
  GEODE_ASSERT(!erased(f));

  // Look up connectivity of neighboring boundary edges, then erase them
  const auto e = faces_[f].neighbors;
  Vector<HalfedgeId,2> near[3]; // (prev,next) for each neighbor boundary edge
  for (int i=0;i<3;i++)
    if (e[i].id<0) {
      const auto& B = boundaries_[-1-e[i].id];
      near[i] = vec(B.prev,B.next);
      unsafe_set_erased(e[i]);
    }

  // Create any new boundary edges and set src and reverse
  HalfedgeId b[3];
  for (int i=0;i<3;i++)
    if (e[i].id>=0) {
      b[i] = unsafe_new_boundary(*this,faces_[f].vertices[i],e[i]);
      const int fi = e[i].id/3;
      faces_.flat[fi].neighbors[e[i].id-3*fi] = b[i];
    }

  // Fix connectivity around each vertex: link together prev/next adjacent edges and update vertex to edge pointers
  for (int i=0, ip=2; i<3; ip=i++) {
    const auto v = faces_[f].vertices[i];
    const auto prev = e[ip].id>=0 ? b[ip] : near[ip].x,
               next = e[i ].id>=0 ? b[i ] : near[i ].y;
    if (e[i].id>=0 || e[ip].id>=0 || prev!=e[i]) {
      unsafe_boundary_link(prev,next);
      vertex_to_edge_[v] = next;
    } else if (vertex_to_edge_[v]==e[ip]) {
      // this vertex is isolated, erase it if requested
      vertex_to_edge_[v] = HalfedgeId();
      if (erase_isolated)
        unsafe_set_erased(v);
    }
  }

  // erase the face
  unsafe_set_erased(f);
}

// Compact the data structure, removing all erased primitives. Returns a tuple of permutations for
// vertices, faces, and boundary halfedges, such that the old primitive i now has index permutation[i].
// Note: non-boundary halfedges don't change order within triangles, so halfedge 3*f+i is now 3*permutation[f]+i
Tuple<Array<int>, Array<int>, Array<int>> TriangleTopology::collect_garbage() {
  Array<int> vertex_permutation(vertex_to_edge_.size()), face_permutation(faces_.size()), boundary_permutation(boundaries_.size());

  int j;

  // first, compact vertex indices (because we only ever decrease ids, we can do this in place)
  j = 0;
  for (int i = 0; i < vertex_to_edge_.size(); ++i) {
    if (!erased(VertexId(i))) {
      vertex_to_edge_.flat[j] = vertex_to_edge_.flat[i];
      vertex_permutation[i] = j;
      j++;
    }
  }
  // discard deleted entries in the back
  vertex_to_edge_.flat.resize(j);
  GEODE_ASSERT(vertex_to_edge_.size() == n_vertices_);

  // now, compact faces
  j = 0;
  for (int i = 0; i < faces_.size(); ++i) {
    if (!erased(FaceId(i))) {
      faces_.flat[j] = faces_.flat[i];
      face_permutation[i] = j;
      j++;
    }
  }
  // discard deleted entries in the back
  faces_.flat.resize(j);
  GEODE_ASSERT(faces_.size() == n_faces_);

  // compact boundaries
  j = 0;
  for (int i = 0; i < boundaries_.size(); ++i) {
    if (!erased(HalfedgeId(-1-i))) {
      boundaries_[j] = boundaries_[i];
      boundary_permutation[i] = j;
      j++;
    }
  }
  boundaries_.resize(j);
  GEODE_ASSERT(boundaries_.size() == n_boundary_edges_);

  // erase boundary free list
  erased_boundaries_ = HalfedgeId();

  // reflect id changes in other arrays
  for (auto& h : vertex_to_edge_.flat) {
    assert(h.id != erased_id);
    h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& f : faces_.flat) {
    assert(f.vertices.x.id != erased_id);
    for (auto& v : f.vertices)
      v = VertexId(vertex_permutation[v.id]);
    for (auto &h : f.neighbors)
      h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& b : boundaries_) {
    assert(b.src.id != erased_id);
    assert(b.src.valid());
    b.src = VertexId(vertex_permutation[b.src.id]);
    b.next = apply_halfedge_permutation(b.next, face_permutation, boundary_permutation);
    b.prev = apply_halfedge_permutation(b.prev, face_permutation, boundary_permutation);
    b.reverse = apply_halfedge_permutation(b.reverse, face_permutation, boundary_permutation);
  }

  return tuple(vertex_permutation, face_permutation, boundary_permutation);
}


static int corner_random_edge_flips(TriangleTopology& mesh, const int attempts, const uint128_t key) {
  int flips = 0;
  if (mesh.n_faces()) {
    const auto random = new_<Random>(key);
    for (int a=0;a<attempts;a++) {
      const HalfedgeId e(random->uniform<int>(0,3*mesh.faces_.size()));
      if (mesh.is_flip_safe(e)) {
        const auto f0 = mesh.face(e),
                   f1 = mesh.face(mesh.reverse(e));
        const auto ef = mesh.unsafe_flip_edge(e);
        GEODE_ASSERT(mesh.face(ef)==f0 && mesh.face(mesh.reverse(ef))==f1);
        flips++;
      }
    }
  }
  return flips;
}

static void corner_random_face_splits(TriangleTopology& mesh, const int splits, const uint128_t key) {
  if (mesh.n_faces()) {
    const auto random = new_<Random>(key);
    for (int a=0;a<splits;a++) {
      const FaceId f(random->uniform<int>(0,mesh.faces_.size()));
      const auto v = mesh.split_face(f);
      GEODE_ASSERT(mesh.face(mesh.halfedge(v))==f);
    }
  }
}

static void corner_mesh_destruction_test(TriangleTopology& mesh, const uint128_t key) {
  const auto random = new_<Random>(key);

  // make two copies, rip one apart using the reordering deletion, and one
  // with the in-place deletion
  auto mesh2 = mesh.copy();

  // with reordering first
  while (mesh2->n_vertices()) {
    GEODE_ASSERT(mesh2->n_faces()==mesh2->faces_.size());
    const int target = random->uniform<int>(0,1+2*mesh2->n_faces());
    if (target<mesh2->n_faces())
      mesh2->erase_face_with_reordering(FaceId(target));
    else
      mesh2->erase_last_vertex_with_reordering();
    mesh2->assert_consistent();
  }

  // in-place (and perform two random garbage collections)
  int gc1_at = random->uniform<int>(mesh.n_vertices()/2, mesh.n_vertices());
  int gc2_at = random->uniform<int>(0, mesh.n_vertices()/2);

  while (mesh.n_vertices()) {

    const bool erase_isolated = random->uniform<int>(0,2);

    // decide whether to erase a face or a halfedge or a vertex
    int choice = random->uniform<int>(0,3);
    if (mesh.n_faces() && choice) {
      if (choice == 1) {
        // find a non-erased face and erase it (there must be one)
        int imax = mesh.faces_.size();
        TriangleTopologyIter<FaceId> target(mesh, FaceId(random->uniform<int>(0, imax)), FaceId(imax));
        while (target.i == target.end) {
          GEODE_ASSERT(target.i != FaceId(0));
          imax = target.i.id;
          target = TriangleTopologyIter<FaceId>(mesh, FaceId(random->uniform<int>(0, imax)), FaceId(imax));
        }
        GEODE_ASSERT(!mesh.erased(*target));
        std::cout << "deleting face " << target.i.id << (erase_isolated ? " and isolated vertices" : "") << std::endl;
        mesh.erase(*target, erase_isolated);
      } else {
        // find a halfedge and erase it (there must be one)
        int imax = mesh.faces_.size();
        TriangleTopologyIter<FaceId> target(mesh, FaceId(random->uniform<int>(0, imax)), FaceId(imax));
        while (target.i == target.end) {
          GEODE_ASSERT(target.i != FaceId(0));
          imax = target.i.id;
          target = TriangleTopologyIter<FaceId>(mesh, FaceId(random->uniform<int>(0, imax)), FaceId(imax));
        }
        GEODE_ASSERT(!mesh.erased(*target));
        int k = random->uniform<int>(0,3);
        std::cout << "deleting halfedge " << target.i.id << ", " << k << (erase_isolated ? " and isolated vertices" : "") << std::endl;
        mesh.erase(mesh.halfedge(*target, k), erase_isolated);
      }
    } else {
      // find a non-erased vertex and erase it (there must be one)
      int imax = mesh.vertex_to_edge_.size();
      TriangleTopologyIter<VertexId> target(mesh, VertexId(random->uniform<int>(0, imax)), VertexId(imax));
      while (target.i == target.end) {
        GEODE_ASSERT(target.i != VertexId(0));
        imax = target.i.id;
        target = TriangleTopologyIter<VertexId>(mesh, VertexId(random->uniform<int>(0, imax)), VertexId(imax));
      }
      GEODE_ASSERT(!mesh.erased(*target));
      std::cout << "deleting vertex " << target.i.id << (erase_isolated ? " and isolated vertices" : "") << std::endl;
      mesh.erase(*target, erase_isolated);
    }
    mesh.assert_consistent();

    if (mesh.n_vertices() <= gc1_at) {
      std::cout << "collect garbage." << std::endl;
      mesh.collect_garbage();
      mesh.assert_consistent();
      if (gc1_at == gc2_at)
        gc1_at = -1;
      else
        gc1_at = gc2_at;
    }
  }
}

}
using namespace geode;

void wrap_corner_mesh() {
  typedef TriangleTopology Self;
  Class<Self>("TriangleTopology")
    .GEODE_INIT()
    .GEODE_METHOD(copy)
    .GEODE_GET(n_vertices)
    .GEODE_GET(n_boundary_edges)
    .GEODE_GET(n_edges)
    .GEODE_GET(n_faces)
    .GEODE_GET(chi)
    .GEODE_METHOD(elements)
    .GEODE_METHOD(has_boundary)
    .GEODE_METHOD(is_manifold)
    .GEODE_METHOD(is_manifold_with_boundary)
    .GEODE_METHOD(has_isolated_vertices)
    .GEODE_METHOD(boundary_loops)
    .GEODE_METHOD(add_vertex)
    .GEODE_METHOD(add_vertices)
    .GEODE_METHOD(add_face)
    .GEODE_METHOD(add_faces)
    .GEODE_METHOD(assert_consistent)
    .GEODE_METHOD(dump_internals)
    .GEODE_METHOD(collect_garbage)
    ;

  // For testing purposes
  GEODE_FUNCTION(corner_random_edge_flips)
  GEODE_FUNCTION(corner_random_face_splits)
  GEODE_FUNCTION(corner_mesh_destruction_test)
}
