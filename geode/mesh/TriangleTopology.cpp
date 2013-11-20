// A corner data structure representing oriented triangle meshes.

#include <geode/mesh/TriangleTopology.h>
#include <geode/python/numpy.h>
#include <geode/array/convert.h>
#include <geode/vector/convert.h>
#include <geode/python/Class.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/Log.h>
#include <boost/dynamic_bitset.hpp>
#include <geode/array/Nested.h>

namespace geode {

using Log::cout;
using std::endl;

GEODE_DEFINE_TYPE(TriangleTopology)
GEODE_DEFINE_TYPE(PropertyStorage)
GEODE_DEFINE_TYPE(MutableTriangleTopology)

static string str_halfedge(HalfedgeId e) {
  return e.valid() ? e.id>=0 ? format("e%d:%d",e.id/3,e.id%3)
                             : format("b%d",-1-e.id)
                   : "e_";
}

TriangleTopology::TriangleTopology()
  : n_vertices_(0)
  , n_faces_(0)
  , n_boundary_edges_(0) {}

TriangleTopology::TriangleTopology(const TriangleTopology& mesh, bool copy)
  : n_vertices_(mesh.n_vertices_)
  , n_faces_(mesh.n_faces_)
  , n_boundary_edges_(mesh.n_boundary_edges_)
  , faces_(copy ? mesh.faces_.copy() : mesh.faces_)
  , vertex_to_edge_(copy ? mesh.vertex_to_edge_.copy() : mesh.vertex_to_edge_)
  , boundaries_(copy ? mesh.boundaries_.copy() : mesh.boundaries_)
  , erased_boundaries_(mesh.erased_boundaries_) {}

TriangleTopology::TriangleTopology(RawArray<const Vector<int,3>> faces)
  : TriangleTopology() {
  const int nodes = faces.size() ? scalar_view(faces).max()+1 : 0;
  internal_add_vertices(nodes);
  internal_add_faces(faces);
  internal_collect_boundary_garbage();
}

TriangleTopology::TriangleTopology(TriangleSoup const &soup)
  : TriangleTopology() {
  internal_add_vertices(soup.nodes());
  internal_add_faces(soup.elements);
  internal_collect_boundary_garbage();
}

TriangleTopology::~TriangleTopology() {}

Ref<TriangleTopology> TriangleTopology::copy() const {
  return new_<TriangleTopology>(*this, true);
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

VertexId TriangleTopology::internal_add_vertex() {
  return internal_add_vertices(1);
}

VertexId TriangleTopology::internal_add_vertices(int n) {
  int id = n_vertices_;
  const_cast<int&>(n_vertices_) += n;
  const_cast_(vertex_to_edge_).const_cast_().flat.resize(vertex_to_edge_.size()+n);
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
  mesh.vertex_to_edge_.const_cast_()[v] = e;
}

// Allocate a fresh boundary edge and set its src and reverse pointers
HalfedgeId TriangleTopology::unsafe_new_boundary(const VertexId src, const HalfedgeId reverse) {
  const_cast_(n_boundary_edges_)++;
  HalfedgeId e;
  if (erased_boundaries_.valid()) {
    e = erased_boundaries_;
    const_cast_(erased_boundaries_) = boundaries_[-1-e.id].next;
  } else {
    const int b = boundaries_.size();
    const_cast_(boundaries_).const_cast_().resize(b+1,false);
    e = HalfedgeId(-1-b);
  }
  boundaries_.const_cast_()[-1-e.id].src = src;
  boundaries_.const_cast_()[-1-e.id].reverse = reverse;
  return e;
}

GEODE_COLD static void add_face_error(const Vector<VertexId,3> v, const char* reason) {
  throw ValueError(format("TriangleTopology::add_face: can't add face (%d,%d,%d)%s",v.x.id,v.y.id,v.z.id,reason));
}

FaceId TriangleTopology::internal_add_face(const Vector<VertexId,3> v) {
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
  const_cast_(n_faces_)++;
  const auto f = const_cast_(faces_).const_cast_().append(FaceInfo());
  faces_.const_cast_()[f].vertices = v;

  // Look up all connectivity we'll need to restructure the mesh.  This includes the reverses
  // r0,r1,r2 of each of e0,e1,e2 (the old halfedges of the triangle), and their future prev/next links.
  // If r0,r1,r2 don't exist yet, they are allocated.
  const auto ve0 = halfedge(v.x),
             ve1 = halfedge(v.y),
             ve2 = halfedge(v.z);
  #define REVERSE(i) \
    const auto r##i = e##i.valid() ? boundaries_[-1-e##i.id].reverse \
                                   : unsafe_new_boundary(v[(i+1)%3],HalfedgeId(3*f.id+i));
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

FaceId TriangleTopology::internal_add_faces(RawArray<const Vector<int,3>> vs) {
  // TODO: We desperately need a batch insertion routine.
  if (vs.empty()) {
    return FaceId();
  } else {
    FaceId first(faces_.size());
    for (auto& v : vs)
      internal_add_face(Vector<VertexId,3>(v));
    return first;
  }
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
    cout << format("    %s: v%d -> %s, p %s, n %s, r %s\n",str_halfedge(e),src(e).id,(valid(n)?format("v%d",src(n).id):"boom"),
                                                           str_halfedge(prev(e)),str_halfedge(n),str_halfedge(reverse(e)));
  }
  cout << endl;
}



MutableTriangleTopology::MutableTriangleTopology()
  : TriangleTopology()
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , max_property_id(100)
{}

MutableTriangleTopology::MutableTriangleTopology(const TriangleTopology& mesh, bool copy)
  : TriangleTopology(mesh, copy)
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , max_property_id(100)
{}

MutableTriangleTopology::MutableTriangleTopology(const MutableTriangleTopology& mesh, bool copy)
  : TriangleTopology(mesh, copy)
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , max_property_id(mesh.max_property_id)
{
  for (auto const &p : mesh.vertex_storage) {
    vertex_storage.set(p.key(), copy ? p.data()->copy() : p.data());
  }
  for (auto const &p : mesh.face_storage) {
    face_storage.set(p.key(), copy ? p.data()->copy() : p.data());
  }
  for (auto const &p : mesh.halfedge_storage) {
    halfedge_storage.set(p.key(), copy ? p.data()->copy() : p.data());
  }
}

MutableTriangleTopology::MutableTriangleTopology(RawArray<const Vector<int,3>> faces)
  : TriangleTopology()
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , max_property_id(100)
{
  add_faces(faces);
}

MutableTriangleTopology::MutableTriangleTopology(TriangleSoup const &soup)
: MutableTriangleTopology(RawArray<const Vector<int,3>>(soup.elements)) {
}

MutableTriangleTopology::~MutableTriangleTopology() {}


HalfedgeId MutableTriangleTopology::unsafe_new_boundary(const VertexId src, const HalfedgeId reverse) {
  HalfedgeId e = TriangleTopology::unsafe_new_boundary(src, reverse);

  // Take care of boundary properties
  // ...

  return e;
}


Ref<MutableTriangleTopology> MutableTriangleTopology::copy() const {
  return new_<MutableTriangleTopology>(*this, true);
}

VertexId MutableTriangleTopology::add_vertex() {
  return add_vertices(1);
}

VertexId MutableTriangleTopology::add_vertices(int n) {
  VertexId id = internal_add_vertices(n);
  for (auto &s : vertex_storage)
    s.data()->grow(n);
  return id;
}

// Add a new face.  If the result would not be manifold, no change is made and ValueError is thrown (TODO: throw a better exception).
FaceId MutableTriangleTopology::add_face(Vector<VertexId,3> v) {
  FaceId id = internal_add_face(v);

  // Take care of halfedges that transitioned from/to boundary
  // ...

  for (auto &s : face_storage)
    s.data()->grow(1);
  for (auto &s : halfedge_storage)
    s.data()->grow(3);

  return id;
}

// Add many new faces (return the first id, new ids are contiguous)
FaceId MutableTriangleTopology::add_faces(RawArray<const Vector<int,3>> vs) {
  FaceId id = internal_add_faces(vs);

  // Take care of halfedges that transitioned from/to boundary
  // ...

  for (auto &s : face_storage)
    s.data()->grow(vs.size());
  for (auto &s : halfedge_storage)
    s.data()->grow(3*vs.size());

  return id;
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

inline static HalfedgeId apply_halfedge_offsets(HalfedgeId h, int face_offset, int boundary_offset) {
  if (h.id != erased_id && h.valid()) {
    if (h.id >= 0) {
      h = HalfedgeId(h.id + face_offset*3);
    } else {
      h = HalfedgeId(h.id - boundary_offset);
    }
  }
  return h;
}

Tuple<int, int, int> MutableTriangleTopology::add(MutableTriangleTopology const &other) {
  // the first indices
  int base_vertex = vertex_to_edge_.size();
  int base_face = faces_.size();
  int base_boundary = boundaries_.size();

  // add things from other to the end
  mutable_vertex_to_edge_.extend(other.vertex_to_edge_);
  mutable_faces_.extend(other.faces_);
  mutable_boundaries_.extend(other.boundaries_);

  // renumber all new primitives

  // reflect id changes in newly added arrays
  for (auto& h : mutable_vertex_to_edge_.flat.slice(base_vertex, vertex_to_edge_.size()))
    h = apply_halfedge_offsets(h, base_face, base_boundary);
  for (auto& f : mutable_faces_.flat.slice(base_face, faces_.size())) {
    if (f.vertices.x.id == erased_id)
      continue;
    for (auto& v : f.vertices)
      v = VertexId(v.id + base_vertex);
    for (auto &h : f.neighbors)
      h = apply_halfedge_offsets(h, base_face, base_boundary);
  }
  for (auto& b : mutable_boundaries_.slice(base_boundary, boundaries_.size())) {
    if (b.src.id == erased_id) {
      // maintain the free list
      b.next = apply_halfedge_offsets(b.next, base_face, base_boundary);
      continue;
    }
    assert(b.src.valid());
    b.src = VertexId(b.src.id + base_vertex);
    b.next = apply_halfedge_offsets(b.next, base_face, base_boundary);
    b.prev = apply_halfedge_offsets(b.prev, base_face, base_boundary);
    b.reverse = apply_halfedge_offsets(b.reverse, base_face, base_boundary);
  }

  // attach the free list
  if (other.erased_boundaries_.valid()) {
    // attach this at the end of our free list
    HalfedgeId bid = HalfedgeId(other.erased_boundaries_.id - base_boundary);

    if (erased_boundaries_.valid()) {
      // go to the end of our free list
      int end = -1-erased_boundaries_.id;
      while (mutable_boundaries_[end].next.valid()) {
        assert(mutable_boundaries_[end].src.id == erased_id);
        end = -1-mutable_boundaries_[end].next.id;
      }
      mutable_boundaries_[end].next = bid;
    } else {
      mutable_erased_boundaries_ = bid;
    }
  }

  // maintain counts
  mutable_n_vertices_ += other.n_vertices_;
  mutable_n_faces_ += other.n_faces_;
  mutable_n_boundary_edges_ += other.n_boundary_edges_;

  // Take care of boundary properties
  // ...

  // Take care to add/update properties. Only properties that exist in *this
  // are considered, and extended if they also exist in other.
  for (auto &s : vertex_storage)
    if (other.vertex_storage.contains(s.key()))
      s.data()->extend(other.vertex_storage.get(s.key()));
    else
      s.data()->grow(other.vertex_storage.get(s.key())->size());
  for (auto &s : face_storage)
    if (other.face_storage.contains(s.key()))
      s.data()->extend(other.face_storage.get(s.key()));
    else
      s.data()->grow(other.face_storage.get(s.key())->size());
  for (auto &s : halfedge_storage)
    if (other.halfedge_storage.contains(s.key()))
      s.data()->extend(other.halfedge_storage.get(s.key()));
    else
      s.data()->grow(other.halfedge_storage.get(s.key())->size());

  return tuple(base_vertex, base_face, base_boundary);
}

void MutableTriangleTopology::split_face(const FaceId f, const VertexId c) {
  GEODE_ASSERT(valid(f) && isolated(c));
  const auto v = faces_[f].vertices;
  const auto n = faces_[f].neighbors;
  const int f_base = faces_.size();
  mutable_n_faces_ += 2;
  mutable_faces_.flat.resize(f_base+2,false);
  const auto fs = vec(f,FaceId(f_base),FaceId(f_base+1));

  #define UPDATE(i) { \
    const int ip = (i+2)%3, in = (i+1)%3; \
    mutable_faces_[fs[i]].vertices.set(v[i],v[in],c); \
    unsafe_set_reverse(fs[i],0,n[i]); \
    mutable_faces_[fs[i]].neighbors[1] = HalfedgeId(3*fs[in].id+2); \
    mutable_faces_[fs[i]].neighbors[2] = HalfedgeId(3*fs[ip].id+1); \
    if (i && mutable_vertex_to_edge_[v[i]].id==3*f.id+i) \
      mutable_vertex_to_edge_[v[i]] = HalfedgeId(3*fs[i].id); }

  UPDATE(0)
  UPDATE(1)
  UPDATE(2)

  #undef UPDATE

  mutable_vertex_to_edge_[c] = halfedge(f,2);

  // Take care to add/update properties on all involved halfedges and faces
  for (auto &s : face_storage)
    s.data()->grow(2);
  for (auto &s : halfedge_storage) {
    s.data()->grow(6);
    // move halfedge data from its original position (in the face f) to
    // where it now belongs: to the reverse of f's original neighbors.
    for (int i = 0; i < 3; ++i) {
      assert(!is_boundary(reverse(n[i])));
      s.data()->swap(3*f.id+i, reverse(n[i]).id);
    }
  }
}

VertexId MutableTriangleTopology::split_face(FaceId f) {
  const auto c = add_vertex();
  split_face(f,c);
  return c;
}

Vector<HalfedgeId,2> MutableTriangleTopology::unsafe_split_halfedge(HalfedgeId h, FaceId nf, VertexId c) {
  assert(!is_boundary(h));

  // remember the old structure
  auto f = face(h);
  auto v = faces_[f].vertices;
  auto n = faces_[f].neighbors;
  auto hf = halfedges(f);

  // which halfedge are we splitting?
  int i = hf.find(h);
  assert(i != -1);
  int iv = (i+1)%3;

  // insert new vertex into existing face
  mutable_faces_[f].vertices[iv] = c;

  // set the vertices of the new face
  mutable_faces_[nf].vertices = vec(c, v[iv], v[(i+2)%3]);

  // fix the halfedge connectivity of the new face (excluding the other face yet to be split)
  unsafe_set_reverse(nf, 1, n[iv]);
  unsafe_set_reverse(nf, 2, hf[iv]);

  // if our face provided the halfedge for the vertex dest(h), we need to change it
  if (face(halfedge(v[iv])) == f) {
    mutable_vertex_to_edge_[v[iv]] = reverse(n[iv]);
  }

  // move halfedge data from its original position (in the face f) to
  // where it now belongs: to the reverse of f's original neighbors. This moves
  // only one of the halfedges.
  for (auto &s : halfedge_storage)
    s.data()->swap(hf[iv].id, reverse(n[iv]).id);

  // return the now dangling halfedges
  return vec(h, halfedge(nf,0));
}

void MutableTriangleTopology::split_edge(HalfedgeId h, VertexId c) {

  // make sure h is not the boundary (if there is a boundary)
  if (is_boundary(h))
    h = reverse(h);

  auto hr = reverse(h);

  // check for a boundary
  bool h_is_boundary = is_boundary(hr);

  // first, grow the storage by the right amount, so we don't do that in pieces
  int n_new_faces = h_is_boundary ? 1 : 2;
  int base_faces = faces_.size();
  mutable_n_faces_ += n_new_faces;
  mutable_faces_.flat.resize(base_faces + n_new_faces);
  for (auto &s : face_storage)
    s.data()->grow(n_new_faces);
  for (auto &s : halfedge_storage)
    s.data()->grow(n_new_faces*3);

  if (h_is_boundary) {
    // Take care of changed boundary edges
    // ...
  }

  auto dangling = unsafe_split_halfedge(h, FaceId(base_faces), c);

  // now deal with reverse(h)
  Vector<HalfedgeId,2> dangling2;
  if (!h_is_boundary)
    dangling2 = unsafe_split_halfedge(hr, FaceId(base_faces+1), c);
  else {
    // deal with a boundary edge hr
    dangling2.x = hr;
    dangling2.y = unsafe_new_boundary(c, HalfedgeId());

    unsafe_boundary_link(dangling2.y, next(hr));
    unsafe_boundary_link(hr, dangling2.y);
  }

  // connect the dangling halfedges
  unsafe_set_reverse(face(dangling.x), dangling.x.id%3, dangling2.y);
  unsafe_set_reverse(face(dangling.y), dangling.y.id%3, dangling2.x);

  // make sure vertex is connected
  mutable_vertex_to_edge_[c] = dangling2.y;
}

VertexId MutableTriangleTopology::split_edge(HalfedgeId e) {
  const auto c = add_vertex();
  split_edge(e,c);
  return c;
}

HalfedgeId MutableTriangleTopology::flip_edge(HalfedgeId e) {
  if (!is_flip_safe(e))
    throw RuntimeError(format("TriangleTopology::flip_edge: edge flip %d is invalid",e.id));
  return unsafe_flip_edge(e);
}

HalfedgeId MutableTriangleTopology::unsafe_flip_edge(HalfedgeId e0) {
  const auto e1 = reverse(e0);
  const auto f0 = face(e0),
             f1 = face(e1);
  const auto n0 = next(e0), p0 = prev(e0),
             n1 = next(e1), p1 = prev(e1),
             rn0 = reverse(n0), rp0 = reverse(p0),
             rn1 = reverse(n1), rp1 = reverse(p1);
  const auto v0 = src(e0), o0 = src(p0),
             v1 = src(e1), o1 = src(p1);
  mutable_faces_[f0].vertices = vec(o0,o1,v1);
  mutable_faces_[f1].vertices = vec(o1,o0,v0);
  mutable_faces_[f0].neighbors.x = HalfedgeId(3*f1.id);
  mutable_faces_[f1].neighbors.x = HalfedgeId(3*f0.id);
  unsafe_set_reverse(f0,1,rp1);
  unsafe_set_reverse(f0,2,rn0);
  unsafe_set_reverse(f1,1,rp0);
  unsafe_set_reverse(f1,2,rn1);
  // Fix vertex to edge links
  auto &ve0 = mutable_vertex_to_edge_[v0],
       &ve1 = mutable_vertex_to_edge_[v1],
       &oe0 = mutable_vertex_to_edge_[o0],
       &oe1 = mutable_vertex_to_edge_[o1];
  if (ve0==e0 || ve0==n1) ve0 = HalfedgeId(3*f1.id+2);
  if (ve1==e1 || ve1==n0) ve1 = HalfedgeId(3*f0.id+2);
  if (oe0==p0) oe0 = HalfedgeId(3*f0.id);
  if (oe1==p1) oe1 = HalfedgeId(3*f1.id);

  // Faces stay the same
  // Take care to add/update properties on all involved halfedges
  for (auto &s : halfedge_storage) {
    // move halfedge data from its original position to where it now belongs:
    // to the reverse of the original reverses (which haven't changed).
    s.data()->swap(n0.id, reverse(rn0).id);
    s.data()->swap(n1.id, reverse(rn1).id);
    s.data()->swap(p0.id, reverse(rp0).id);
    s.data()->swap(p1.id, reverse(rp1).id);
  }

  return HalfedgeId(3*f0.id);
}

void MutableTriangleTopology::erase_last_vertex_with_reordering() {
  const VertexId v(vertex_to_edge_.size()-1);
  // erase all incident faces
  while (!isolated(v)) {
    erase_face_with_reordering(face(reverse(halfedge(v))));
  }

  // Remove the vertex
  mutable_vertex_to_edge_.flat.pop();
  mutable_n_vertices_--;

  // take care of vertex storage
  for (auto &s : vertex_storage)
    s.data()->grow(-1);
}

void MutableTriangleTopology::erase_face_with_reordering(const FaceId f) {
  GEODE_ASSERT(f.valid());
  GEODE_ASSERT(f.id != erased_id);

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
      b[i] = unsafe_new_boundary(faces_[f].vertices[i],e[i]);
      const int fi = e[i].id/3;
      mutable_faces_.flat[fi].neighbors[e[i].id-3*fi] = b[i];
    }

  // Fix connectivity around each vertex: link together prev/next adjacent edges and update vertex to edge pointers
  for (int i=0;i<3;i++) {
    const int ip = i?i-1:2;
    const auto v = faces_[f].vertices[i];
    const auto prev = e[ip].id>=0 ? b[ip] : near[ip].x,
               next = e[i ].id>=0 ? b[i ] : near[i ].y;
    if (e[i].id>=0 || e[ip].id>=0 || prev!=e[i]) {
      unsafe_boundary_link(prev,next);
      mutable_vertex_to_edge_[v] = next;
    } else if (vertex_to_edge_[v]==e[ip])
      mutable_vertex_to_edge_[v] = HalfedgeId();
  }

  // Rename the last face to f
  const FaceId f1(faces_.size()-1);
  if (f.id<f1.id) {
    assert(!erased(f1));
    const auto I = faces_[f1];
    mutable_faces_[f].vertices = I.vertices;
    for (int i=0;i<3;i++) {
      unsafe_set_reverse(f,i,I.neighbors[i]);
      if (mutable_vertex_to_edge_[I.vertices[i]].id==3*f1.id+i)
        mutable_vertex_to_edge_[I.vertices[i]].id = 3*f.id+i;
    }
  }

  // Take care of boundary properties
  // ...

  // Remove the erased face
  mutable_faces_.flat.pop();
  mutable_n_faces_--;

  // update the properties on halfedges and faces
  for (auto &s : face_storage) {
    s.data()->swap(f.id, f1.id);
    s.data()->grow(-1);
  }
  for (auto &s : halfedge_storage) {
    // move halfedge data from its original position (in the face f) to
    // where it now belongs: to the reverse of f's original neighbors.
    for (int i = 0; i < 3; ++i) {
      s.data()->swap(3*f.id+i, 3*f1.id+i);
    }
    s.data()->grow(-3);
  }
}

void MutableTriangleTopology::permute_vertices(RawArray<const int> permutation, bool check) {
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
  mutable_vertex_to_edge_.flat = new_vertex_to_edge;

  // The other arrays can be modified in place
  for (auto& f : mutable_faces_.flat)
    if (f.vertices.x.id!=erased_id)
      for (auto& v : f.vertices)
        v = VertexId(permutation[v.id]);
  for (auto& b : mutable_boundaries_)
    if (b.src.id!=erased_id)
      b.src = VertexId(permutation[b.src.id]);

  // permute properties
  for (auto &s : vertex_storage)
    s.data()->apply_permutation(permutation);
}

// erase the given vertex. erases all incident faces. If erase_isolated is true, also erase other vertices that are now isolated.
void MutableTriangleTopology::erase(VertexId id, bool erase_isolated) {
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
void MutableTriangleTopology::erase(HalfedgeId id, bool erase_isolated) {
  auto he = faces(id);
  for (auto h : he) {
    if (valid(h)) {
      erase(h, erase_isolated);
    }
  }
}

// erase the given face. If erase_isolated is true, also erases incident vertices that are now isolated.
void MutableTriangleTopology::erase(FaceId f, bool erase_isolated) {
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
      b[i] = unsafe_new_boundary(faces_[f].vertices[i],e[i]);
      const int fi = e[i].id/3;
      mutable_faces_.flat[fi].neighbors[e[i].id-3*fi] = b[i];
    }

  // Fix connectivity around each vertex: link together prev/next adjacent edges and update vertex to edge pointers
  for (int i=0, ip=2; i<3; ip=i++) {
    const auto v = faces_[f].vertices[i];
    const auto prev = e[ip].id>=0 ? b[ip] : near[ip].x,
               next = e[i ].id>=0 ? b[i ] : near[i ].y;
    if (e[i].id>=0 || e[ip].id>=0 || prev!=e[i]) {
      unsafe_boundary_link(prev,next);
      mutable_vertex_to_edge_[v] = next;
    } else if (vertex_to_edge_[v]==e[ip]) {
      // this vertex is isolated, erase it if requested
      mutable_vertex_to_edge_[v] = HalfedgeId();
      if (erase_isolated)
        unsafe_set_erased(v);
    }
  }

  // add/update properties on boundaries
  // ...

  // erase the face
  unsafe_set_erased(f);
}

// Compact the data structure, removing all erased primitives. Returns a tuple of permutations for
// vertices, faces, and boundary halfedges, such that the old primitive i now has index permutation[i].
// Note: non-boundary halfedges don't change order within triangles, so halfedge 3*f+i is now 3*permutation[f]+i
Tuple<Array<int>,Array<int>,Array<int>> MutableTriangleTopology::collect_garbage() {
  Array<int> vertex_permutation(vertex_to_edge_.size()), face_permutation(faces_.size()), boundary_permutation(boundaries_.size());

  // first, compact vertex indices (because we only ever decrease ids, we can do this in place)
  int j = 0;
  for (int i = 0; i < vertex_to_edge_.size(); ++i) {
    if (!erased(VertexId(i))) {
      mutable_vertex_to_edge_.flat[j] = vertex_to_edge_.flat[i];
      vertex_permutation[i] = j;
      j++;
    } else {
      vertex_permutation[i] = -1;
    }
  }
  // discard deleted entries in the back
  mutable_vertex_to_edge_.flat.resize(j);
  GEODE_ASSERT(vertex_to_edge_.size() == n_vertices_);

  // now, compact faces
  j = 0;
  for (int i = 0; i < faces_.size(); ++i) {
    if (!erased(FaceId(i))) {
      mutable_faces_.flat[j] = faces_.flat[i];
      face_permutation[i] = j;
      j++;
    } else {
      face_permutation[i] = -1;
    }
  }
  // discard deleted entries in the back
  mutable_faces_.flat.resize(j);
  GEODE_ASSERT(faces_.size() == n_faces_);

  // compact boundaries
  j = 0;
  for (int i = 0; i < boundaries_.size(); ++i) {
    if (!erased(HalfedgeId(-1-i))) {
      mutable_boundaries_[j] = mutable_boundaries_[i];
      boundary_permutation[i] = j;
      j++;
    } else {
      boundary_permutation[i] = -1;
    }
  }
  mutable_boundaries_.resize(j);
  GEODE_ASSERT(boundaries_.size() == n_boundary_edges_);

  // erase boundary free list
  mutable_erased_boundaries_ = HalfedgeId();

  // reflect id changes in other arrays
  for (auto& h : mutable_vertex_to_edge_.flat) {
    assert(h.id != erased_id);
    h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& f : mutable_faces_.flat) {
    assert(f.vertices.x.id != erased_id);
    for (auto& v : f.vertices)
      v = VertexId(vertex_permutation[v.id]);
    for (auto &h : f.neighbors)
      h = apply_halfedge_permutation(h, face_permutation, boundary_permutation);
  }
  for (auto& b : mutable_boundaries_) {
    assert(b.src.id != erased_id);
    assert(b.src.valid());
    b.src = VertexId(vertex_permutation[b.src.id]);
    b.next = apply_halfedge_permutation(b.next, face_permutation, boundary_permutation);
    b.prev = apply_halfedge_permutation(b.prev, face_permutation, boundary_permutation);
    b.reverse = apply_halfedge_permutation(b.reverse, face_permutation, boundary_permutation);
  }

  for (auto &s : vertex_storage)
    s.data()->apply_permutation(vertex_permutation);
  for (auto &s : face_storage)
    s.data()->apply_permutation(face_permutation);

  // make a halfedge permutation
  Array<int> halfedge_permutation(face_permutation.size()*3);
  for (int i = 0; i < face_permutation.size(); ++i) {
    halfedge_permutation[3*i+0] = 3*face_permutation[i];
    halfedge_permutation[3*i+1] = 3*face_permutation[i]+1;
    halfedge_permutation[3*i+2] = 3*face_permutation[i]+2;
  }
  for (auto &s : halfedge_storage)
    s.data()->apply_permutation(halfedge_permutation);

  // also apply permutation to boundary fields
  // ...

  return tuple(vertex_permutation, face_permutation, boundary_permutation);
}

Array<int> TriangleTopology::internal_collect_boundary_garbage() {
  // Compact boundaries
  int j = 0;
  Array<int> boundary_permutation(boundaries_.size());
  for (int i=0;i<boundaries_.size();i++) {
    if (!erased(HalfedgeId(-1-i))) {
      const_cast_(boundaries_[j]) = boundaries_[i];
      boundary_permutation[i] = j;
      j++;
    } else {
      boundary_permutation[i] = -1;
    }
  }
  const_cast_(boundaries_).const_cast_().resize(j);
  GEODE_ASSERT(boundaries_.size() == n_boundary_edges_);

  // Erase boundary free list
  const_cast_(erased_boundaries_) = HalfedgeId();

  // Apply boundary permutation
  for (int i=0;i<boundaries_.size();i++) {
    auto& B = boundaries_[i];
    const auto b = HalfedgeId(-1-i);
    const auto r = B.reverse;
    const auto f = FaceId(r.id/3);
    unsafe_set_reverse(f,r.id-3*f.id,b);
    const_cast_(vertex_to_edge_[B.src]) = b;
    const_cast_(B.prev) = HalfedgeId(-1-boundary_permutation[-1-B.prev.id]);
    const_cast_(B.next) = HalfedgeId(-1-boundary_permutation[-1-B.next.id]);
  }
  return boundary_permutation;
}

Array<int> MutableTriangleTopology::collect_boundary_garbage() {
  const auto p = internal_collect_boundary_garbage();
  // Once we have boundary fields, we will need to apply p to them
  return p;
}

bool TriangleTopology::is_garbage_collected() const {
  return n_vertices_ == vertex_to_edge_.size() &&
         n_faces_ == faces_.size() &&
         n_boundary_edges_ == boundaries_.size();
}

#ifdef GEODE_PYTHON

#define ADDPROP(primitive, ...) \
    case NumpyScalar<__VA_ARGS__>::value: { \
      auto pid = to_python(add_##primitive##_property<__VA_ARGS__>(id));\
      return pid; \
    }

#define ADDSPROPS(primitive) \
    ADDPROP(primitive,bool)\
    ADDPROP(primitive,char)\
    ADDPROP(primitive,unsigned char)\
    ADDPROP(primitive,short)\
    ADDPROP(primitive,unsigned short)\
    ADDPROP(primitive,int)\
    ADDPROP(primitive,unsigned int)\
    ADDPROP(primitive,long)\
    ADDPROP(primitive,unsigned long)\
    ADDPROP(primitive,long long)\
    ADDPROP(primitive,unsigned long long)\
    ADDPROP(primitive,float)\
    ADDPROP(primitive,double)

#define ADDVPROPS(primitive,d) \
    ADDPROP(primitive,Vector<bool,d>)\
    ADDPROP(primitive,Vector<char,d>)\
    ADDPROP(primitive,Vector<unsigned char,d>)\
    ADDPROP(primitive,Vector<short,d>)\
    ADDPROP(primitive,Vector<unsigned short,d>)\
    ADDPROP(primitive,Vector<int,d>)\
    ADDPROP(primitive,Vector<unsigned int,d>)\
    ADDPROP(primitive,Vector<long,d>)\
    ADDPROP(primitive,Vector<unsigned long,d>)\
    ADDPROP(primitive,Vector<long long,d>)\
    ADDPROP(primitive,Vector<unsigned long long,d>)\
    ADDPROP(primitive,Vector<float,d>)\
    ADDPROP(primitive,Vector<double,d>)

#define MAKE_PY_PROPERTY(primitive) \
PyObject *MutableTriangleTopology::add_##primitive##_property_py(PyObject *object, int id) { \
  PyArray_Descr* dtype;\
  if (!PyArray_DescrConverter(object,&dtype))\
    return NULL;\
  const Ref<> save = steal_ref(*(PyObject*)dtype);\
  int type = dtype->type_num;\
  \
  if (dtype->subarray == NULL) {\
    switch (type) {\
      ADDSPROPS(primitive)\
      default:\
        Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));\
        throw TypeError(format("properties of type %s unavailable from Python",from_python<const char*>(s)));\
    }\
  } else {\
    type = dtype->subarray->base->type_num;\
    auto shape = from_python<Array<const int>>(dtype->subarray->shape);\
    switch (shape.size()) {\
      case 1:\
        switch (shape[0]) {\
          case 2:\
            switch (type) {\
              ADDVPROPS(primitive,2)\
              default:\
                Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));\
                throw TypeError(format("properties of type %s unavailable from Python",from_python<const char*>(s)));\
            }\
          case 3:\
            switch (type) {\
              ADDVPROPS(primitive,3)\
              default:\
                Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));\
                throw TypeError(format("properties of type %s unavailable from Python",from_python<const char*>(s)));\
            }\
          case 4:\
            switch (type) {\
              ADDVPROPS(primitive,4)\
              default:\
                Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));\
                throw TypeError(format("properties of type %s unavailable from Python",from_python<const char*>(s)));\
            }\
        }\
      default:\
        Ref<PyObject> s = steal_ref_check(PyObject_Str((PyObject*)dtype));\
        throw TypeError(format("properties of type %s unavailable from Python",from_python<const char*>(s)));\
    }\
  }\
}

MAKE_PY_PROPERTY(vertex)
MAKE_PY_PROPERTY(face)
MAKE_PY_PROPERTY(halfedge)

bool MutableTriangleTopology::has_property_py(PyPropertyId const &id) const {

  GEODE_ASSERT(id.fancy == false);

#define CASE(...) \
  if (id.type_id == typeid(__VA_ARGS__).name()) { \
    switch (id.id_type) {\
      case PyPropertyId::idVertex: return has_vertex_property(PropertyId<__VA_ARGS__,VertexId,false>(id.id)); \
      case PyPropertyId::idFace: return has_face_property(PropertyId<__VA_ARGS__,FaceId,false>(id.id)); \
      case PyPropertyId::idHalfedge: return has_halfedge_property(PropertyId<__VA_ARGS__,HalfedgeId,false>(id.id)); \
    } \
  }

  CASE(bool)
  CASE(char)
  CASE(unsigned char)
  CASE(short)
  CASE(unsigned short)
  CASE(int)
  CASE(unsigned int)
  CASE(long)
  CASE(unsigned long)
  CASE(long long)
  CASE(unsigned long long)
  CASE(float)
  CASE(double)
  CASE(Vector<bool,2>)
  CASE(Vector<char,2>)
  CASE(Vector<unsigned char,2>)
  CASE(Vector<short,2>)
  CASE(Vector<unsigned short,2>)
  CASE(Vector<int,2>)
  CASE(Vector<unsigned int,2>)
  CASE(Vector<long,2>)
  CASE(Vector<unsigned long,2>)
  CASE(Vector<long long,2>)
  CASE(Vector<unsigned long long,2>)
  CASE(Vector<float,2>)
  CASE(Vector<double,2>)
  CASE(Vector<bool,3>)
  CASE(Vector<char,3>)
  CASE(Vector<unsigned char,3>)
  CASE(Vector<short,3>)
  CASE(Vector<unsigned short,3>)
  CASE(Vector<int,3>)
  CASE(Vector<unsigned int,3>)
  CASE(Vector<long,3>)
  CASE(Vector<unsigned long,3>)
  CASE(Vector<long long,3>)
  CASE(Vector<unsigned long long,3>)
  CASE(Vector<float,3>)
  CASE(Vector<double,3>)
  CASE(Vector<bool,4>)
  CASE(Vector<char,4>)
  CASE(Vector<unsigned char,4>)
  CASE(Vector<short,4>)
  CASE(Vector<unsigned short,4>)
  CASE(Vector<int,4>)
  CASE(Vector<unsigned int,4>)
  CASE(Vector<long,4>)
  CASE(Vector<unsigned long,4>)
  CASE(Vector<long long,4>)
  CASE(Vector<unsigned long long,4>)
  CASE(Vector<float,4>)
  CASE(Vector<double,4>)

  #undef CASE

  throw std::runtime_error(format("Can't handle python conversion of properties of type %s", id.type_id));
  return false;
}

void MutableTriangleTopology::remove_property_py(PyPropertyId const &id) {

  GEODE_ASSERT(id.fancy == false);

#define CASE(...) \
  if (id.type_id == typeid(__VA_ARGS__).name()) { \
    switch (id.id_type) {\
      case PyPropertyId::idVertex: remove_vertex_property(PropertyId<__VA_ARGS__,VertexId,false>(id.id)); \
      case PyPropertyId::idFace: remove_face_property(PropertyId<__VA_ARGS__,FaceId,false>(id.id)); \
      case PyPropertyId::idHalfedge: remove_halfedge_property(PropertyId<__VA_ARGS__,HalfedgeId,false>(id.id)); \
    } \
    return;\
  }

  CASE(bool)
  CASE(char)
  CASE(unsigned char)
  CASE(short)
  CASE(unsigned short)
  CASE(int)
  CASE(unsigned int)
  CASE(long)
  CASE(unsigned long)
  CASE(long long)
  CASE(unsigned long long)
  CASE(float)
  CASE(double)
  CASE(Vector<bool,2>)
  CASE(Vector<char,2>)
  CASE(Vector<unsigned char,2>)
  CASE(Vector<short,2>)
  CASE(Vector<unsigned short,2>)
  CASE(Vector<int,2>)
  CASE(Vector<unsigned int,2>)
  CASE(Vector<long,2>)
  CASE(Vector<unsigned long,2>)
  CASE(Vector<long long,2>)
  CASE(Vector<unsigned long long,2>)
  CASE(Vector<float,2>)
  CASE(Vector<double,2>)
  CASE(Vector<bool,3>)
  CASE(Vector<char,3>)
  CASE(Vector<unsigned char,3>)
  CASE(Vector<short,3>)
  CASE(Vector<unsigned short,3>)
  CASE(Vector<int,3>)
  CASE(Vector<unsigned int,3>)
  CASE(Vector<long,3>)
  CASE(Vector<unsigned long,3>)
  CASE(Vector<long long,3>)
  CASE(Vector<unsigned long long,3>)
  CASE(Vector<float,3>)
  CASE(Vector<double,3>)
  CASE(Vector<bool,4>)
  CASE(Vector<char,4>)
  CASE(Vector<unsigned char,4>)
  CASE(Vector<short,4>)
  CASE(Vector<unsigned short,4>)
  CASE(Vector<int,4>)
  CASE(Vector<unsigned int,4>)
  CASE(Vector<long,4>)
  CASE(Vector<unsigned long,4>)
  CASE(Vector<long long,4>)
  CASE(Vector<unsigned long long,4>)
  CASE(Vector<float,4>)
  CASE(Vector<double,4>)

#undef CASE

  throw std::runtime_error(format("Can't handle python conversion of properties of type %s", id.type_id));
}


PyObject *MutableTriangleTopology::property_py(PyPropertyId const &id) {

  GEODE_ASSERT(id.fancy == false);

#define CASE(...) \
  if (id.type_id == typeid(__VA_ARGS__).name()) { \
    switch (id.id_type) {\
      case PyPropertyId::idVertex: return to_python(property(PropertyId<__VA_ARGS__,VertexId,false>(id.id))); \
      case PyPropertyId::idFace: return to_python(property(PropertyId<__VA_ARGS__,FaceId,false>(id.id))); \
      case PyPropertyId::idHalfedge: return to_python(property(PropertyId<__VA_ARGS__,HalfedgeId,false>(id.id))); \
    } \
  }

  CASE(bool)
  CASE(char)
  CASE(unsigned char)
  CASE(short)
  CASE(unsigned short)
  CASE(int)
  CASE(unsigned int)
  CASE(long)
  CASE(unsigned long)
  CASE(long long)
  CASE(unsigned long long)
  CASE(float)
  CASE(double)
  CASE(Vector<bool,2>)
  CASE(Vector<char,2>)
  CASE(Vector<unsigned char,2>)
  CASE(Vector<short,2>)
  CASE(Vector<unsigned short,2>)
  CASE(Vector<int,2>)
  CASE(Vector<unsigned int,2>)
  CASE(Vector<long,2>)
  CASE(Vector<unsigned long,2>)
  CASE(Vector<long long,2>)
  CASE(Vector<unsigned long long,2>)
  CASE(Vector<float,2>)
  CASE(Vector<double,2>)
  CASE(Vector<bool,3>)
  CASE(Vector<char,3>)
  CASE(Vector<unsigned char,3>)
  CASE(Vector<short,3>)
  CASE(Vector<unsigned short,3>)
  CASE(Vector<int,3>)
  CASE(Vector<unsigned int,3>)
  CASE(Vector<long,3>)
  CASE(Vector<unsigned long,3>)
  CASE(Vector<long long,3>)
  CASE(Vector<unsigned long long,3>)
  CASE(Vector<float,3>)
  CASE(Vector<double,3>)
  CASE(Vector<bool,4>)
  CASE(Vector<char,4>)
  CASE(Vector<unsigned char,4>)
  CASE(Vector<short,4>)
  CASE(Vector<unsigned short,4>)
  CASE(Vector<int,4>)
  CASE(Vector<unsigned int,4>)
  CASE(Vector<long,4>)
  CASE(Vector<unsigned long,4>)
  CASE(Vector<long long,4>)
  CASE(Vector<unsigned long long,4>)
  CASE(Vector<float,4>)
  CASE(Vector<double,4>)

#undef CASE

  throw std::runtime_error(format("Can't handle python conversion of properties of type %s", id.type_id));
  return NULL;
}

#endif

static int corner_random_edge_flips(MutableTriangleTopology& mesh, const int attempts, const uint128_t key) {
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

static void corner_random_face_splits(MutableTriangleTopology& mesh, const int splits, const uint128_t key) {
  if (mesh.n_faces()) {
    const auto random = new_<Random>(key);
    for (int a=0;a<splits;a++) {
      const FaceId f(random->uniform<int>(0,mesh.faces_.size()));
      const auto v = mesh.split_face(f);
      GEODE_ASSERT(mesh.face(mesh.halfedge(v))==f);
    }
    for (int a=0;a<splits;a++) {
      const HalfedgeId h(random->uniform<int>(0,mesh.faces_.size()*3));
      const auto v = mesh.split_edge(h);
      GEODE_ASSERT(mesh.halfedge(v)==mesh.reverse(h));
    }
  }
}

static void corner_mesh_destruction_test(MutableTriangleTopology& mesh, const uint128_t key) {
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

template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIncoming>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyOutgoing>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<VertexId>>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<FaceId>>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<HalfedgeId>>);

}
using namespace geode;

void wrap_corner_mesh() {
  {
    typedef TriangleTopology Self;
    Class<Self>("TriangleTopology")
      .GEODE_INIT(const TriangleSoup&)
      .GEODE_METHOD(copy)
      .GEODE_GET(n_vertices)
      .GEODE_GET(n_boundary_edges)
      .GEODE_GET(n_edges)
      .GEODE_GET(n_faces)
      .GEODE_GET(chi)
      .GEODE_OVERLOADED_METHOD(HalfedgeId(Self::*)(VertexId)const, halfedge)
      .GEODE_METHOD(prev)
      .GEODE_METHOD(next)
      .GEODE_METHOD(src)
      .GEODE_METHOD(dst)
      .GEODE_METHOD(face)
      .GEODE_METHOD(left)
      .GEODE_METHOD(right)
      .method("face_vertices", static_cast<Vector<VertexId,3>(Self::*)(FaceId)const>(&Self::vertices))
      .method("face_halfedges", static_cast<Vector<HalfedgeId,3>(Self::*)(FaceId)const>(&Self::halfedges))
      .method("halfedge_vertices", static_cast<Vector<VertexId,2>(Self::*)(HalfedgeId)const>(&Self::vertices))
      .method("face_faces", static_cast<Vector<FaceId,3>(Self::*)(FaceId)const>(&Self::faces))
      .method("halfedge_faces", static_cast<Vector<FaceId,2>(Self::*)(HalfedgeId)const>(&Self::faces))
      .GEODE_METHOD(outgoing)
      .GEODE_METHOD(incoming)
      .GEODE_METHOD(vertex_one_ring)
      .GEODE_METHOD(incident_faces)
      .GEODE_OVERLOADED_METHOD_2(HalfedgeId(Self::*)(VertexId, VertexId)const, "halfedge_between", halfedge)
      .GEODE_METHOD(common_halfedge)
      .GEODE_METHOD(elements)
      .GEODE_METHOD(degree)
      .GEODE_METHOD(has_boundary)
      .GEODE_METHOD(is_manifold)
      .GEODE_METHOD(is_manifold_with_boundary)
      .GEODE_METHOD(has_isolated_vertices)
      .GEODE_METHOD(boundary_loops)
      .GEODE_METHOD(assert_consistent)
      .GEODE_METHOD(dump_internals)
      .GEODE_METHOD(all_vertices)
      .GEODE_METHOD(all_faces)
      .GEODE_METHOD(all_halfedges)
      .GEODE_METHOD(all_boundary_edges)
      .GEODE_METHOD(all_interior_halfedges)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<VertexId>>(Self::*)() const, vertices)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<FaceId>>(Self::*)() const, faces)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<HalfedgeId>>(Self::*)() const, halfedges)
      .GEODE_METHOD(boundary_edges)
      .GEODE_METHOD(interior_halfedges)
      .GEODE_METHOD(is_garbage_collected)
      ;
  }
  {
    typedef MutableTriangleTopology Self;
    Class<Self>("MutableTriangleTopology")
      .GEODE_INIT()
      .GEODE_METHOD(copy)
      .GEODE_METHOD(add_vertex)
      .GEODE_METHOD(add_vertices)
      .GEODE_METHOD(add_face)
      .GEODE_METHOD(add_faces)
      .GEODE_OVERLOADED_METHOD_2(void(Self::*)(FaceId,bool), "erase_face", erase)
      .GEODE_OVERLOADED_METHOD_2(void(Self::*)(VertexId,bool), "erase_vertex", erase)
      .GEODE_OVERLOADED_METHOD_2(void(Self::*)(HalfedgeId,bool), "erase_halfedge", erase)
      .GEODE_METHOD(collect_garbage)
      .GEODE_METHOD(collect_boundary_garbage)
      .GEODE_METHOD_2("add_vertex_property", add_vertex_property_py)
      .GEODE_METHOD_2("add_face_property", add_face_property_py)
      .GEODE_METHOD_2("add_halfedge_property", add_halfedge_property_py)
      .GEODE_METHOD_2("has_property", has_property_py)
      .GEODE_METHOD_2("remove_property", remove_property_py)
      .GEODE_METHOD_2("property", property_py)
      .GEODE_METHOD(permute_vertices)
      ;
  }
  // For testing purposes
  GEODE_FUNCTION(corner_random_edge_flips)
  GEODE_FUNCTION(corner_random_face_splits)
  GEODE_FUNCTION(corner_mesh_destruction_test)

  GEODE_PYTHON_RANGE(TriangleTopologyIncoming, "IncomingHalfedgeIter")
  GEODE_PYTHON_RANGE(TriangleTopologyOutgoing, "OutgoingHalfedgeIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<VertexId>, "SkippingVertexIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<FaceId>, "SkippingFaceIter")
  GEODE_PYTHON_RANGE(TriangleTopologyIter<HalfedgeId>, "SkippingHalfedgeIter")
}
