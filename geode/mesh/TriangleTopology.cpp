// A corner data structure representing oriented triangle meshes.

#include <geode/mesh/TriangleTopology.h>
#include <geode/mesh/SegmentSoup.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/array/convert.h>
#include <geode/array/Nested.h>
#include <geode/array/permute.h>
#include <geode/python/numpy.h>
#include <geode/python/Class.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/Log.h>
#include <geode/vector/convert.h>
#include <geode/structure/UnionFind.h>
namespace geode {

using Log::cout;
using std::endl;
typedef real T;
typedef Vector<T,2> TV2;
typedef Vector<T,3> TV3;

GEODE_DEFINE_TYPE(TriangleTopology)
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

TriangleTopology::TriangleTopology(const TriangleSoup& soup)
  : TriangleTopology() {
  internal_add_vertices(soup.nodes());
  internal_add_faces(soup.elements);
  internal_collect_boundary_garbage();
}

TriangleTopology::~TriangleTopology() {}

Ref<TriangleTopology> TriangleTopology::copy() const {
  return new_<TriangleTopology>(*this,true);
}

Ref<MutableTriangleTopology> TriangleTopology::mutate() const {
  return new_<MutableTriangleTopology>(*this,true);
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

VertexId TriangleTopology::common_vertex(FaceId f, HalfedgeId e) const {
  if (f.valid() && e.valid()) {
    VertexId s = src(e);
    VertexId d = dst(e);
    for (const auto v : vertices(f)) {
      if (d == v) {
        return d;
      }
      if (s == v) {
        return s;
      }
    }
  }
  return VertexId();
}

VertexId TriangleTopology::common_vertex(FaceId f0, FaceId f1) const {
  if (f0.valid() && f1.valid()) {
    auto v1 = vertices(f1);
    for (const auto v : vertices(f0)) {
      if (v1.contains(v))
        return v;
    }
  }
  return VertexId();
}

VertexId TriangleTopology::internal_add_vertex() {
  return internal_add_vertices(1);
}

VertexId TriangleTopology::internal_add_vertices(int n) {
  GEODE_ASSERT(n >= 0);
  int id = vertex_to_edge_.size(); // This doesn't attempt to recycle erased vertex ids
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
  } else
    e = HalfedgeId(-1-const_cast_(boundaries_).append(uninit));
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

void TriangleTopology::assert_consistent(bool check_for_double_halfedges) const {
  // Check simple vertex properties
  int actual_vertices = 0;
  for (const auto v : vertices()) {
    GEODE_ASSERT(valid(v));
    actual_vertices++;
    const auto e = halfedge(v);
    if (e.valid())
      GEODE_ASSERT(src(e)==v);
  }
  GEODE_ASSERT(actual_vertices==n_vertices());

  // Check simple face properties
  int actual_faces = 0;
  for (const auto f : faces()) {
    GEODE_ASSERT(valid(f));
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
    GEODE_ASSERT(valid(e));
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
  if (check_for_double_halfedges) {
    Hashtable<Vector<VertexId,2>> pairs;
    for (const auto e : halfedges())
      GEODE_ASSERT(pairs.set(vertices(e)));
  }

  // Check that all halfedges are reachable by swinging around their source vertices, and that
  // boundary vertices point to boundary halfedges.
  {
    int count = 0;
    vector<bool> seen(boundaries_.size()+3*faces_.size());
    for (const auto v : vertices())
      if (!isolated(v)) {
        bool boundary = false;
        for (const auto e : outgoing(v)) {
          GEODE_ASSERT(src(e)==v);
          const int i = boundaries_.size()+e.id;
          if (!seen[i]) {
            seen[i] = true;
            count++;
          }
          boundary |= is_boundary(e);
        }
        GEODE_ASSERT(boundary==is_boundary(v));
      }
    GEODE_ASSERT(count==2*n_edges());
  }

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
  Array<Vector<int,3>> tris(n_faces(),uninit);
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
  for (const auto start : boundary_edges())
    for (auto e=start;;) {
      e = left(e);
      if (e==start)
        break;
      if (is_boundary(e)) // There are two boundary halfedges at the same vertex, which is bad.
        return false;
    }
  return true;
}

Nested<FaceId> TriangleTopology::surface_components(VertexId v) const {
  auto incident = incident_faces(v);

  // map face handles to indices in incident
  Hashtable<FaceId, int> fmap;
  for (auto it = incident.begin(); it != incident.end(); ++it) {
    fmap[*it] = (int)(it-incident.begin());
  }

  UnionFind union_find((int)incident.size());
  for (auto ohe : outgoing(v)) {
    if (is_boundary(ohe) || is_boundary(reverse(ohe)))
      continue;
    auto fs = faces(ohe);
    assert(fmap.contains(fs.x));
    assert(fmap.contains(fs.y));
    union_find.merge(fmap[fs.x], fmap[fs.y]);
  }

  // spit out connected components
  Hashtable<int, Array<FaceId>> components;
  for (auto f : incident) {
    components[union_find.find(fmap[f])].append(f);
  }
  Nested<FaceId, false> result;
  for (auto c : components) {
    result.append(c.y);
  }
  return result.freeze();
}

bool TriangleTopology::has_isolated_vertices() const {
  for (const auto v : vertices())
    if (isolated(v))
      return true;
  return false;
}

int TriangleTopology::degree(VertexId v) const {
  GEODE_ASSERT(valid(v));
  int degree = 0;
  for (GEODE_UNUSED auto _ : outgoing(v))
    degree++;
  return degree;
}

Nested<HalfedgeId> TriangleTopology::boundary_loops() const {
  Nested<HalfedgeId> loops;
  vector<bool> seen(boundaries_.size());
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
  , next_field_id(100)
{}

MutableTriangleTopology::MutableTriangleTopology(const TriangleTopology& mesh, bool copy)
  : TriangleTopology(mesh,copy)
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , next_field_id(100)
{}

MutableTriangleTopology::MutableTriangleTopology(const MutableTriangleTopology& mesh, bool copy)
  : TriangleTopology(mesh,copy)
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , id_to_vertex_field(mesh.id_to_vertex_field)
  , id_to_face_field(mesh.id_to_face_field)
  , id_to_halfedge_field(mesh.id_to_halfedge_field)
  , next_field_id(mesh.next_field_id)
{
  for (const auto& p : mesh.vertex_fields)     vertex_fields.push_back(copy ? p.copy() : p);
  for (const auto& p : mesh.face_fields)         face_fields.push_back(copy ? p.copy() : p);
  for (const auto& p : mesh.halfedge_fields) halfedge_fields.push_back(copy ? p.copy() : p);
}

MutableTriangleTopology::MutableTriangleTopology(RawArray<const Vector<int,3>> faces)
  : TriangleTopology(faces)
  , mutable_n_vertices_(const_cast_(n_vertices_))
  , mutable_n_faces_(const_cast_(n_faces_))
  , mutable_n_boundary_edges_(const_cast_(n_boundary_edges_))
  , mutable_faces_(const_cast_(faces_).const_cast_())
  , mutable_vertex_to_edge_(const_cast_(vertex_to_edge_).const_cast_())
  , mutable_boundaries_(const_cast_(boundaries_).const_cast_())
  , mutable_erased_boundaries_(const_cast_(erased_boundaries_))
  , next_field_id(100)
{}

MutableTriangleTopology::MutableTriangleTopology(TriangleSoup const &soup)
: MutableTriangleTopology(RawArray<const Vector<int,3>>(soup.elements)) {
}

MutableTriangleTopology::~MutableTriangleTopology() {}


HalfedgeId MutableTriangleTopology::unsafe_new_boundary(const VertexId src, const HalfedgeId reverse) {
  HalfedgeId e = TriangleTopology::unsafe_new_boundary(src, reverse);

  // Take care of boundary fields
  // ...

  return e;
}


Ref<MutableTriangleTopology> MutableTriangleTopology::copy() const {
  return new_<MutableTriangleTopology>(*this, true);
}

VertexId MutableTriangleTopology::add_vertex() {
  return add_vertices(1);
}

VertexId MutableTriangleTopology::copy_vertex(VertexId v) {
  VertexId id = add_vertex();

  // copy all vertex fields to the new vertex
  for (auto f : vertex_fields) {
    f.copy(id.idx(), v.idx());
  }

  return id;
}

VertexId MutableTriangleTopology::add_vertices(int n) {
  VertexId id = internal_add_vertices(n);
  for (auto& s : vertex_fields)
    s.extend(n);
  return id;
}

FaceId MutableTriangleTopology::add_face(Vector<VertexId,3> v) {
  const FaceId id = internal_add_face(v);

  // Take care of halfedges that transitioned from/to boundary
  for (auto& s : face_fields)
    s.extend(1);
  for (auto& s : halfedge_fields)
    s.extend(3);

  return id;
}

// Add many new faces (return the first id, new ids are contiguous)
FaceId MutableTriangleTopology::add_faces(RawArray<const Vector<int,3>> vs) {
  FaceId id = internal_add_faces(vs);

  // Take care of halfedges that transitioned from/to boundary
  for (auto& s : face_fields)
    s.extend(vs.size());
  for (auto& s : halfedge_fields)
    s.extend(3*vs.size());

  return id;
}

static inline HalfedgeId permute_halfedge(HalfedgeId h, RawArray<const int> face_permutation, RawArray<const int> boundary_permutation) {
  assert(h.id != erased_id);
  if (!h.valid())
    return h;
  if (h.id >= 0)
    return HalfedgeId(3*face_permutation[h.id/3] + h.id%3);
  else
    return HalfedgeId(-1-boundary_permutation[-1-h.id]);
}

static inline HalfedgeId offset_halfedge(HalfedgeId h, int face_offset, int boundary_offset) {
  if (h.id != erased_id && h.valid()) {
    if (h.id >= 0) {
      h = HalfedgeId(h.id + face_offset*3);
    } else {
      h = HalfedgeId(h.id - boundary_offset);
    }
  }
  return h;
}

Vector<int,3> MutableTriangleTopology::add(const MutableTriangleTopology& other) {
  // Record first indices
  const int base_vertex = vertex_to_edge_.size();
  const int base_face = faces_.size();
  const int base_boundary = boundaries_.size();

  // Add things from other to the end
  mutable_vertex_to_edge_.extend(other.vertex_to_edge_.flat);
  mutable_faces_.extend(other.faces_.flat);
  mutable_boundaries_.extend(other.boundaries_);

  // Renumber all new primitives

  // Reflect id changes in newly added arrays
  for (auto& h : mutable_vertex_to_edge_.flat.slice(base_vertex, vertex_to_edge_.size()))
    h = offset_halfedge(h, base_face, base_boundary);
  for (auto& f : mutable_faces_.flat.slice(base_face, faces_.size())) {
    if (f.vertices.x.id == erased_id)
      continue;
    for (auto& v : f.vertices)
      v = VertexId(v.id + base_vertex);
    for (auto &h : f.neighbors)
      h = offset_halfedge(h, base_face, base_boundary);
  }
  for (auto& b : mutable_boundaries_.slice(base_boundary, boundaries_.size())) {
    if (b.src.id == erased_id) {
      // Maintain the free list
      b.next = offset_halfedge(b.next, base_face, base_boundary);
      continue;
    }
    assert(b.src.valid());
    b.src = VertexId(b.src.id + base_vertex);
    b.next = offset_halfedge(b.next, base_face, base_boundary);
    b.prev = offset_halfedge(b.prev, base_face, base_boundary);
    b.reverse = offset_halfedge(b.reverse, base_face, base_boundary);
  }

  // Attach the free list
  if (other.erased_boundaries_.valid()) {
    // Attach this at the end of our free list
    HalfedgeId bid = HalfedgeId(other.erased_boundaries_.id - base_boundary);

    if (erased_boundaries_.valid()) {
      // Go to the end of our free list
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

  // Maintain counts
  mutable_n_vertices_ += other.n_vertices_;
  mutable_n_faces_ += other.n_faces_;
  mutable_n_boundary_edges_ += other.n_boundary_edges_;

  // Take care of boundary fields
  // ...

  // Add/update fields. Only fields that exist in *this
  // are considered, and extended if they also exist in other.
  #define FIELD(prim, size_expr) \
    for (auto& it : id_to_##prim##_field) { \
      const int d = it.y; \
      const int s = other.id_to_##prim##_field.get_default(it.x,-1); \
      if (s >= 0) \
        prim##_fields[d].extend(other.prim##_fields[s]); \
      else \
        prim##_fields[d].extend(size_expr); \
    }
  FIELD(vertex, other.vertex_to_edge_.size())
  FIELD(face,   other.faces_.size())
  FIELD(halfedge, 3*other.faces_.size())
  #undef FIELD

  // Return index offsets
  return vec(base_vertex,base_face,base_boundary);
}


void MutableTriangleTopology::flip() {

  // boundary
  for (auto he : boundary_edges()) {
    // set src=dst (safely compute dst without using prev/next, which may be broken)
    unsafe_set_src(he, src(reverse(he)));
    // swap prev and next
    std::swap(mutable_boundaries_[-1-he.id].next, mutable_boundaries_[-1-he.id].prev);
  }

  // interior
  for (auto f : faces()) {
    auto &fi = mutable_faces_[f];
    // swap vertices 0 and 1 in each face, and swap neighors 1 and 2.
    fi.vertices = fi.vertices.yxz();
    fi.neighbors = fi.neighbors.xzy();
    // adjust neighbors' neighbors to point back to the swapped neighbors
    assert(reverse(fi.neighbors[1]) == halfedge(f,2));
    assert(reverse(fi.neighbors[2]) == halfedge(f,1));
    for (int i : vec(1,2)) {
      auto he = fi.neighbors[i];
      if (is_boundary(he)) {
        mutable_boundaries_[-1-he.id].reverse = halfedge(f,i);
      } else {
        mutable_faces_[face(he)].neighbors[face_index(he)] = halfedge(f,i);
      }
    }
    // swap halfedge fields
    for (auto &field : halfedge_fields) {
      field.swap(halfedge(f,1).id, halfedge(f,2).id);
    }
  }

  // fix vertex outgoing halfedges
  for (auto v : vertices()) {
    if (halfedge(v).valid() && src(halfedge(v)) != v) {
      if (is_boundary(halfedge(v))) {
        // we should now be dst(halfedge(v))
        assert(v == dst(halfedge(v)));
        unsafe_set_halfedge(v, next(halfedge(v)));
      } else {
        // reconstruct which edge belongs to us from our position in the triangle
        auto f = face(halfedge(v));
        auto fi = faces_[f];
        int i = fi.vertices.find(v);
        assert(i != -1);
        unsafe_set_halfedge(v, halfedge(f,i));
      }
      assert(src(halfedge(v))==v);
    }
  }
}

Ref<MutableTriangleTopology> MutableTriangleTopology::flipped() const {
  auto mesh = copy();
  mesh->flip();
  return mesh;
}

Tuple<Ref<MutableTriangleTopology>,
      Field<VertexId, VertexId>,
      Field<FaceId, FaceId>> MutableTriangleTopology::extract(RawArray<FaceId> const &faces) {

  Ref<MutableTriangleTopology> result = new_<MutableTriangleTopology>();

  Hashtable<VertexId, VertexId> old_to_new_vertices;
  Hashtable<FaceId, FaceId> old_to_new_faces;

  for (FaceId face : faces) {
    GEODE_ASSERT(valid(face));
    Vector<VertexId, 3> verts, oldverts = vertices(face);

    // find or add vertices for this face
    for (int i = 0; i < 3; ++i) {
      VertexId oldv = oldverts[i];
      VertexId &newv = verts[i];

      if (old_to_new_vertices.contains(oldv)) {
        newv = old_to_new_vertices[oldv];
      } else {
        newv = result->add_vertex();
        old_to_new_vertices.insert(oldv, newv);
      }
    }

    // add the face
    old_to_new_faces.insert(face, result->add_face(verts));
  }

  // copy all fields
  result->id_to_vertex_field = id_to_vertex_field;
  result->id_to_face_field = id_to_face_field;
  result->id_to_halfedge_field = id_to_halfedge_field;

  // add field data
  for (auto a : vertex_fields) {
    result->vertex_fields.push_back(UntypedArray::empty_like(a, result->n_vertices()));
    auto field = result->vertex_fields.back();

    // copy necessary data
    for (auto v : old_to_new_vertices) {
      VertexId vold = v.x;
      VertexId vnew = v.y;
      field.copy_from(vnew.idx(), a, vold.idx());
    }
  }

  for (auto a : face_fields) {
    result->face_fields.push_back(UntypedArray::empty_like(a, result->n_faces()));
    auto field = result->face_fields.back();

    // copy necessary data
    for (auto f : old_to_new_faces) {
      FaceId fold = f.x;
      FaceId fnew = f.y;
      field.copy_from(fnew.idx(), a, fold.idx());
    }
  }

  for (auto a : halfedge_fields) {
    result->halfedge_fields.push_back(UntypedArray::empty_like(a, 3*result->n_faces()));
    auto field = result->halfedge_fields.back();

    // copy necessary data
    for (auto f : old_to_new_faces) {
      HalfedgeId fold = HalfedgeId(3*f.x.id);
      HalfedgeId fnew = HalfedgeId(3*f.y.id);
      field.copy_from(fnew.idx(), a, fold.idx());
      field.copy_from(fnew.idx()+1, a, fold.idx()+1);
      field.copy_from(fnew.idx()+2, a, fold.idx()+2);
    }
  }

  auto new_to_old_vertices = result->create_compatible_vertex_field<VertexId>();
  for (auto v : old_to_new_vertices) {
    new_to_old_vertices[v.y] = v.x;
  }

  auto new_to_old_faces = result->create_compatible_face_field<FaceId>();
  for (auto v : old_to_new_faces) {
    new_to_old_faces[v.y] = v.x;
  }

  return tuple(result, new_to_old_vertices, new_to_old_faces);
}


Array<VertexId> MutableTriangleTopology::split_nonmanifold_vertex(VertexId vi) {
  auto components = surface_components(vi);

  Array<VertexId> verts;
  if (components.size() < 2) {
    verts.append(vi);
    return verts;
  }

  for (int i = 0; i < (int) components.size(); ++i) {
    VertexId v = (i > 0) ? copy_vertex(vi) : vi;

    Vector<HalfedgeId, 2> boundaries;
    for (auto f : components[i]) {

      // replace vi with v in f
      if (i > 0)
        unsafe_replace_vertex(f, vi, v);

      // remember boundary edges (we'll need to patch stuff up later)
      // replace vi with v in src field of boundary edges
      for (auto he : halfedges(f)) {
        if (src(he) == v && is_boundary(reverse(he))) {
          GEODE_ASSERT(!boundaries.x.valid());
          boundaries.x = reverse(he);
        } else if (dst(he) == v && is_boundary(reverse(he))) {
          GEODE_ASSERT(!boundaries.y.valid());
          boundaries.y = reverse(he);
          if (i > 0)
            unsafe_set_src(reverse(he), v);
        }
      }
    }
    verts.append(v);

    GEODE_ASSERT(boundaries.x.valid() && boundaries.y.valid());

    // make sure the new vertex has a boundary edge as its halfedge
    unsafe_set_halfedge(v, boundaries.y);

    // patch up boundaries (this component has only two boundary edges, connect
    // them (src is already fixed)
    unsafe_boundary_link(boundaries.x, boundaries.y);
  }

  return verts;
}

Nested<VertexId> MutableTriangleTopology::split_nonmanifold_vertices() {
  Hashtable<VertexId> verts;
  for (auto e : boundary_edges()) {
    verts.set(src(e));
  }
  Nested<VertexId,false> result;
  for (auto v : verts) {
    auto newv = split_nonmanifold_vertex(v);
    if (newv.size() > 1)
      result.append(newv);
  }
  return result.freeze();
}

Vector<HalfedgeId, 2> MutableTriangleTopology::split_along_edge(HalfedgeId he) {
  auto re = reverse(he);

  auto b_he = unsafe_new_boundary(dst(he), he);
  auto b_re = unsafe_new_boundary(dst(re), re);

  unsafe_boundary_link(b_he, b_re);
  unsafe_boundary_link(b_re, b_he);

  unsafe_set_reverse(face(he), face_index(he), b_he);
  unsafe_set_reverse(face(re), face_index(re), b_re);

  if (!is_boundary(dst(he)))
    unsafe_set_halfedge(dst(he), b_he);
  if (!is_boundary(dst(re)))
    unsafe_set_halfedge(dst(re), b_re);

  return vec(b_he, b_re);
}

void MutableTriangleTopology::split_face(const FaceId f, const VertexId c) {
  GEODE_ASSERT(valid(f) && isolated(c));
  const auto v = faces_[f].vertices;
  const auto n = faces_[f].neighbors;
  mutable_n_faces_ += 2;
  const int f_base = mutable_faces_.flat.extend(2,uninit);
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

  // Add/update fields on all involved halfedges and faces
  for (auto& s : face_fields)
    s.extend(2);
  for (auto& s : halfedge_fields) {
    s.extend(6);
    // Move halfedge data from its original position (in the face f) to
    // where it now belongs: to the reverse of f's original neighbors.
    for (int i=0;i<3;i++) {
      assert(!is_boundary(reverse(n[i])));
      s.swap(3*f.id+i,reverse(n[i]).id);
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

  // Remember the old structure
  auto f = face(h);
  auto v = faces_[f].vertices;
  auto n = faces_[f].neighbors;
  auto hf = halfedges(f);

  // Which halfedge are we splitting?
  int i = hf.find(h);
  assert(i != -1);
  int iv = (i+1)%3;

  // Insert new vertex into existing face
  mutable_faces_[f].vertices[iv] = c;

  // Set the vertices of the new face
  mutable_faces_[nf].vertices = vec(c,v[iv],v[(i+2)%3]);

  // Fix the halfedge connectivity of the new face (excluding the other face yet to be split)
  unsafe_set_reverse(nf, 1, n[iv]);
  unsafe_set_reverse(nf, 2, hf[iv]);

  // If our face provided the halfedge for the vertex dest(h), we need to change it
  if (face(halfedge(v[iv])) == f) {
    mutable_vertex_to_edge_[v[iv]] = reverse(n[iv]);
  }

  // Move halfedge data from its original position (in the face f) to
  // where it now belongs: to the reverse of f's original neighbors. This moves
  // only one of the halfedges.
  for (auto& s : halfedge_fields)
    s.swap(hf[iv].id,reverse(n[iv]).id);

  // return the now dangling halfedges
  return vec(h,halfedge(nf,0));
}

void MutableTriangleTopology::split_edge(HalfedgeId h, VertexId c) {
  // Make sure h is not the boundary (if there is a boundary)
  if (is_boundary(h))
    h = reverse(h);

  const auto hr = reverse(h);

  // check for a boundary
  const bool h_is_boundary = is_boundary(hr);

  // first, grow the storage by the right amount, so we don't do that in pieces
  const int n_new_faces = h_is_boundary ? 1 : 2;
  const int base_faces = faces_.size();
  mutable_n_faces_ += n_new_faces;
  mutable_faces_.flat.resize(base_faces + n_new_faces);
  for (auto& s : face_fields)
    s.extend(n_new_faces);
  for (auto& s : halfedge_fields)
    s.extend(3*n_new_faces);

  if (h_is_boundary) {
    // Take care of changed boundary edges
    // ...
  }

  const auto dangling = unsafe_split_halfedge(h,FaceId(base_faces),c);

  // Now deal with reverse(h)
  Vector<HalfedgeId,2> dangling2;
  if (!h_is_boundary)
    dangling2 = unsafe_split_halfedge(hr,FaceId(base_faces+1),c);
  else {
    // Deal with a boundary edge hr
    dangling2.x = hr;
    dangling2.y = unsafe_new_boundary(c, HalfedgeId());

    unsafe_boundary_link(dangling2.y, next(hr));
    unsafe_boundary_link(hr, dangling2.y);
  }

  // Connect the dangling halfedges
  unsafe_set_reverse(face(dangling.x), dangling.x.id%3, dangling2.y);
  unsafe_set_reverse(face(dangling.y), dangling.y.id%3, dangling2.x);

  // Make sure vertex is connected
  mutable_vertex_to_edge_[c] = dangling2.y;
}

VertexId MutableTriangleTopology::split_edge(HalfedgeId e) {
  const auto c = add_vertex();
  split_edge(e,c);
  return c;
}

bool MutableTriangleTopology::is_collapse_safe(HalfedgeId h) const {
  GEODE_ASSERT(valid(h));
  const auto o = reverse(h);
  const auto v0 = src(h),
             v1 = dst(h);

  // If v0 and v1 are on different boundaries, we can't do this
  if (is_boundary(v0) && is_boundary(v1) &&
      !is_boundary(h) && !is_boundary(o))
    return false;

  // Can't snip off an isolated vl or vr
  if ((is_boundary(reverse(next(h))) && is_boundary(reverse(prev(h)))) ||
      (is_boundary(reverse(next(o))) && is_boundary(reverse(prev(o)))))
    return false;

  // Look up left and right vertices
  const auto vl = is_boundary(h) ? VertexId() : dst(next(h)),
             vr = is_boundary(o) ? VertexId() : dst(next(o));

  // This only happens in temporarily invalid situations, such as if
  // split_along_edge is called and not cleaned up.  No good can come of it.
  if (vl==vr)
    return false;

  // One-rings of v0 and v1 cannot intersect, otherwise we'll collapse a
  // triangle-shaped tunnel
  Hashtable<VertexId> covered;
  for (auto oh: outgoing(v0)) {
    auto v = dst(oh);
    if (v != vl && v != vr)
      covered.set(v);
  }
  for (auto oh: outgoing(v1)) {
    auto v = dst(oh);
    if (covered.contains(v))
      return false;
  }

  return true;
}

void MutableTriangleTopology::unsafe_collapse(HalfedgeId h) {
  GEODE_ASSERT(valid(h));

  HalfedgeId o = reverse(h);

  VertexId v0 = src(h);
  VertexId v1 = dst(h);

  VertexId vl;
  if (!is_boundary(h))
    vl = dst(next(h));

  VertexId vr;
  if (!is_boundary(o))
    vr = dst(next(o));

  // We need to choose a new outgoing halfedge for v1 if the current one is going to be erased or if v1 becomes a boundary halfedge
  HalfedgeId new_v1_he; // If set, v1's halfedge will be changed to this after the collapse is finished
  // Hitting an empty leaf in this conditional will cause v1's halfedge to remain unchanged since new_v1_he will remain invalid
  if(is_boundary(o)) {
    // In this case, o will be the only erased halfedge
    // If v1 was using o we need to switch
    if(halfedge(v1) == o) {
      new_v1_he = next(o); // next(o) will be a boundary halfedge and after we collapse v0 into v1 it will start at v1
    }
    // else, we don't need to change anything so leave new_v1_he invalid
  }
  else if(is_boundary(h)) {
    // halfedge(v1) must be a boundary edge (probably next(h) but possibly a different halfedge if there are disjoint boundary curves at v1)
    assert(is_boundary(halfedge(v1)));
    // We don't need to change v1's halfedge since it can't be h and h is the only erased boundary halfedge
  }
  else {
    // If we get here, we aren't erasing any boundary halfedges
    if(is_boundary(halfedge(v1))) {
      // v1's halfedge is on the boundary so we aren't going to erase it
    }
    else if(is_boundary(halfedge(v0))) {
      new_v1_he = halfedge(v0); // Combined vertex will now be on boundary so reuse halfedge(v0) to get a boundary halfedge with the right source
    }
    else {
      if(halfedge(v1) == o || dst(halfedge(v1)) == vl) {
        new_v1_he = left(o); // Can use v1vr which we must exist and isn't going to be erased
      }
    }
  }

  // if vl used vlv0 as outgoing halfedge, change it to something else
  if (vl.valid() && dst(halfedge(vl)) == v0) {
    unsafe_set_halfedge(vl, left(halfedge(vl)));
  }

  // if vr used vrv1 as outgoing halfedge, change it to something else
  if (vr.valid() && dst(halfedge(vr)) == v1) {
    unsafe_set_halfedge(vr, left(halfedge(vr)));
  }

  // replace v0 with v1 in all faces
  for (auto f : incident_faces(v0))
    unsafe_replace_vertex(f, v0, v1);

  // replace v0 with v1 in all boundary edge src entries
  // Since we might have pointers from multiple disjoint boundary curves we have to check all outgoing halfedges of v0
  if (is_boundary(v0)) for(const auto he : outgoing(v0)) {
    if(is_boundary(he))
      unsafe_set_src(he, v1);
  }

  // connect reverses: v0vl -- vlv1, vrv0 -- v1vr
  // (if v0vl and vlv1 or vrv0 and v1vr are boundaries, don't connect them,
  // delete them. Patch up boundary, make vl or vr isolated vertices)
  if (vl.valid()) {
    auto v0vl = reverse(prev(h));
    auto vlv1 = reverse(next(h));

    if (is_boundary(v0vl) && is_boundary(vlv1)) {
      unsafe_boundary_link(prev(v0vl), next(vlv1));
      unsafe_set_erased(v0vl);
      unsafe_set_erased(vlv1);
      if (halfedge(vl) == vlv1)
        unsafe_set_halfedge(vl,HalfedgeId());
    } else {
      unsafe_set_reverse(v0vl, vlv1);
    }
  }

  if (vr.valid()) {
    auto vrv0 = reverse(next(o));
    auto v1vr = reverse(prev(o));

    if (is_boundary(vrv0) && is_boundary(v1vr)) {
      unsafe_boundary_link(prev(v1vr), next(vrv0));
      unsafe_set_erased(v1vr);
      unsafe_set_erased(vrv0);
      if (halfedge(vr) == vrv0)
        unsafe_set_halfedge(vr,HalfedgeId());
    } else {
      unsafe_set_reverse(vrv0, v1vr);
    }
  }

  if (vl.valid())
    // delete face incident to h
    unsafe_set_erased(face(h));
  else {
    unsafe_boundary_link(prev(h), next(h));
    unsafe_set_erased(h);
  }

  if (vr.valid())
    // delete face incident to o
    unsafe_set_erased(face(o));
  else {
    unsafe_set_src(next(o), v1);
    unsafe_boundary_link(prev(o), next(o));
    unsafe_set_erased(o);
  }

  // delete v0
  unsafe_set_erased(v0);

  // If we need to update halfedge for v1, do so here
  if(new_v1_he.valid()) {
    assert(erased(halfedge(v1)) || (is_boundary(new_v1_he) && !is_boundary(halfedge(v1))));
    assert(!erased(new_v1_he));
    unsafe_set_halfedge(v1, new_v1_he);
    assert(src(new_v1_he) == v1);
  }

  assert(!erased(halfedge(v1)));

  assert_consistent(true);
}

void MutableTriangleTopology::collapse(HalfedgeId h) {
  if (!is_collapse_safe(h))
    throw RuntimeError(format("TriangleTopology::collapse: halfedge collapse %d [%d,%d] is invalid",
                              h.id,src(h).id,dst(h).id));
  unsafe_collapse(h);
}

HalfedgeId MutableTriangleTopology::flip_edge(HalfedgeId e) {
  if (!is_flip_safe(e))
    throw RuntimeError(format("TriangleTopology::flip_edge: edge flip %d [%d,%d] is invalid",
                              e.id,src(e).id,dst(e).id));
  return unsafe_flip_edge(e);
}

bool MutableTriangleTopology::unsafe_replace_vertex(FaceId id, VertexId oldv, VertexId newv) {
  for (int i = 0; i < 3; ++i) {
    if (faces_[id].vertices[i] == oldv) {
      mutable_faces_[id].vertices[i] = newv;
      return true;
    }
  }
  return false;
}

bool MutableTriangleTopology::ensure_boundary_halfedge(VertexId v) {
  bool is_boundary_now = is_boundary(halfedge(v));
  for (auto he : outgoing(v)) {
    if (is_boundary(he) && !is_boundary_now) {
      unsafe_set_halfedge(v, he);
      return true;
    }
  }
  return false;
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
  // Take care to add/update fields on all involved halfedges
  for (auto& s : halfedge_fields) {
    // Move halfedge data from its original position to where it now belongs:
    // to the reverse of the original reverses (which haven't changed).
    s.swap(n0.id,reverse(rn0).id);
    s.swap(n1.id,reverse(rn1).id);
    s.swap(p0.id,reverse(rp0).id);
    s.swap(p1.id,reverse(rp1).id);
  }

  return HalfedgeId(3*f0.id);
}

void MutableTriangleTopology::erase_last_vertex_with_reordering() {
  const VertexId v(vertex_to_edge_.size()-1);
  // Erase all incident faces
  while (!isolated(v))
    erase_face_with_reordering(face(reverse(halfedge(v))));

  // Remove the vertex
  mutable_vertex_to_edge_.flat.pop();
  mutable_n_vertices_--;

  // Take care of vertex storage
  for (auto& s : vertex_fields)
    s.extend(-1);
}

void MutableTriangleTopology::erase_isolated_vertices() {
  auto marked = create_compatible_vertex_field<bool>();
  for (auto f : faces()) {
    for (auto v : vertices(f)) {
      marked[v] = true;
    }
  }

  for (auto v : vertices()) {
    if (!marked[v]) {
      erase(v);
    }
  }
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

  // Take care of boundary fields
  // ...

  // Remove the erased face
  mutable_faces_.flat.pop();
  mutable_n_faces_--;

  // Update the fields on halfedges and faces
  for (auto& s : face_fields) {
    s.swap(f.id,f1.id);
    s.extend(-1);
  }
  for (auto& s : halfedge_fields) {
    // Move halfedge data from its original position (in the face f) to
    // where it now belongs: to the reverse of f's original neighbors.
    for (int i=0;i<3;i++)
      s.swap(3*f.id+i,3*f1.id+i);
    s.extend(-3);
  }
}

void MutableTriangleTopology::permute_vertices(RawArray<const int> permutation, bool check) {
  GEODE_ASSERT(n_vertices()==permutation.size());
  GEODE_ASSERT(n_vertices()==vertex_to_edge_.size()); // Require no erased vertices

  // Permute vertex_to_edge_ out of place
  Array<HalfedgeId> new_vertex_to_edge(vertex_to_edge_.size(),uninit);
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

  // Permute fields
  Array<char> work;
  for (auto& s : vertex_fields)
    inplace_partial_permute(s,permutation,work);
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

  // Add/update fields on boundaries
  // ...

  // Erase the face
  unsafe_set_erased(f);
}

// Compact the data structure, removing all erased primitives. Returns a tuple of permutations for
// vertices, faces, and boundary halfedges, such that the old primitive i now has index permutation[i].
// Note: non-boundary halfedges don't change order within triangles, so halfedge 3*f+i is now 3*permutation[f]+i
Vector<Array<int>,3> MutableTriangleTopology::collect_garbage() {
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
    h = permute_halfedge(h, face_permutation, boundary_permutation);
  }
  for (auto& f : mutable_faces_.flat) {
    assert(f.vertices.x.id != erased_id);
    for (auto& v : f.vertices)
      v = VertexId(vertex_permutation[v.id]);
    for (auto &h : f.neighbors)
      h = permute_halfedge(h, face_permutation, boundary_permutation);
  }
  for (auto& b : mutable_boundaries_) {
    assert(b.src.id != erased_id);
    assert(b.src.valid());
    b.src = VertexId(vertex_permutation[b.src.id]);
    b.next = permute_halfedge(b.next, face_permutation, boundary_permutation);
    b.prev = permute_halfedge(b.prev, face_permutation, boundary_permutation);
    b.reverse = permute_halfedge(b.reverse, face_permutation, boundary_permutation);
  }

  Array<char> work;
  for (auto& s : vertex_fields)
    inplace_partial_permute(s,vertex_permutation,work);
  for (auto& s : face_fields)
    inplace_partial_permute(s,face_permutation,work);
  for (auto& s : halfedge_fields)
    inplace_partial_permute(s,face_permutation,work,3);

  // Also apply permutation to boundary fields
  // ...

  return vec(vertex_permutation,face_permutation,boundary_permutation);
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

Array<VertexId> TriangleTopology::vertex_one_ring(VertexId v) const {
  GEODE_ASSERT(valid(v));
  Array<VertexId> result;
  for (auto h : outgoing(v)) {
    result.append(dst(h));
  }
  return result;
}

Array<FaceId> TriangleTopology::incident_faces(VertexId v) const {
  GEODE_ASSERT(valid(v));
  Array<FaceId> faces;
  for (const auto h : outgoing(v)) {
    const auto f = face(h);
    if (f.valid())
      faces.append(f);
  }
  return faces;
}

int TriangleTopology::valence(VertexId v) const {
  GEODE_ASSERT(valid(v));
  int count=0;
  for (const auto h : outgoing(v)) {
    const auto f = face(h);
    if (f.valid())
      count++;
  }
  return count;
}

void remove_field_helper(Hashtable<int,int>& id_to_field, vector<UntypedArray>& fields, const int id) {
  const int i = id_to_field.get_default(id,-1);
  if (i >= 0) {
    id_to_field.erase(id);
    const int j = int(fields.size())-1;
    if (i < j) {
      fields[i] = fields.back();
      for (auto& h : id_to_field)
        if (h.y == j) {
          h.y = i;
          break;
        }
    }
    fields.pop_back();
  }
}

#ifdef GEODE_PYTHON

#define ADD_FIELD_HELPER(prim, ...) \
  case NumpyScalar<__VA_ARGS__>::value: \
    return to_python_ref(add_##prim##_field<__VA_ARGS__>(id));

#define ADD_FIELD(prim, d, ...) \
  ADD_FIELD_HELPER(prim,mpl::if_c<d==0,__VA_ARGS__,Vector<__VA_ARGS__,d>>::type)

#define ADD_FIELDS(prim,d) \
  ADD_FIELD(prim,d,bool) \
  ADD_FIELD(prim,d,char) \
  ADD_FIELD(prim,d,unsigned char) \
  ADD_FIELD(prim,d,short) \
  ADD_FIELD(prim,d,unsigned short) \
  ADD_FIELD(prim,d,int) \
  ADD_FIELD(prim,d,unsigned int) \
  ADD_FIELD(prim,d,long) \
  ADD_FIELD(prim,d,unsigned long) \
  ADD_FIELD(prim,d,long long) \
  ADD_FIELD(prim,d,unsigned long long) \
  ADD_FIELD(prim,d,float) \
  ADD_FIELD(prim,d,double)

#define MAKE_PY_FIELD(prim, Prim) \
  Ref<> MutableTriangleTopology::add_##prim##_field_py(PyObject* object, const int id) { \
    PyArray_Descr* dtype; \
    if (!PyArray_DescrConverter(object,&dtype)) \
      throw_python_error();                          \
    const Ref<> save = steal_ref(*(PyObject*)dtype); \
    if (!dtype->subarray) \
      switch (dtype->type_num) { ADD_FIELDS(prim,0) } \
    else { \
      const int subtype = dtype->subarray->base->type_num; \
      const auto shape = from_python<Array<const int>>(dtype->subarray->shape); \
      if (shape.size() == 1) { \
        switch (shape[0]) { \
          case 2: switch (subtype) { ADD_FIELDS(prim,2) } break; \
          case 3: switch (subtype) { ADD_FIELDS(prim,3) } break; \
          case 4: switch (subtype) { ADD_FIELDS(prim,4) } break; \
        } \
      } \
    } \
    const auto s = steal_ref_check(PyObject_Str((PyObject*)dtype)); \
    throw TypeError(format("Fields of type %s unavailable from Python",from_python<const char*>(s))); \
  }\
  bool MutableTriangleTopology::has_##prim##_field_py(int id) const {\
    return has_field_py(new_<PyFieldId>(PyFieldId::Prim, id));\
  }\
  void MutableTriangleTopology::remove_##prim##_field_py(int id) {\
    return remove_field_py(new_<PyFieldId>(PyFieldId::Prim, id));\
  }\
  Ref<> MutableTriangleTopology::prim##_field_py(int id) {   \
    return field_py(new_<PyFieldId>(PyFieldId::Prim, id));\
  }

MAKE_PY_FIELD(vertex, Vertex)
MAKE_PY_FIELD(face, Face)
MAKE_PY_FIELD(halfedge, Halfedge)

bool MutableTriangleTopology::has_field_py(const PyFieldId& id) const {
  // Check if the field id exists
  const auto& id_to_field = id.prim == PyFieldId::Vertex ? id_to_vertex_field
                          : id.prim == PyFieldId::Face   ? id_to_face_field
                                                         : id_to_halfedge_field;
  const int i = id_to_field.get_default(id.id,-1);
  if (i < 0)
    return false;

  // Check if the type matches
  const auto& fields = id.prim == PyFieldId::Vertex ? vertex_fields
                     : id.prim == PyFieldId::Face   ? face_fields
                                                    : halfedge_fields;
  const auto& field = fields[i];
  return &(field.type()) == id.type;
}

void MutableTriangleTopology::remove_field_py(const PyFieldId& id) {
  auto& id_to_field = id.prim == PyFieldId::Vertex ? id_to_vertex_field
                    : id.prim == PyFieldId::Face   ? id_to_face_field
                                                   : id_to_halfedge_field;
  auto& fields = id.prim == PyFieldId::Vertex ? vertex_fields
               : id.prim == PyFieldId::Face   ? face_fields
                                              : halfedge_fields;
  remove_field_helper(id_to_field,fields,id.id);
}

Ref<> MutableTriangleTopology::field_py(const PyFieldId& id) {
  const auto& id_to_field = id.prim == PyFieldId::Vertex ? id_to_vertex_field
                          : id.prim == PyFieldId::Face   ? id_to_face_field
                                                         : id_to_halfedge_field;
  const auto& fields = id.prim == PyFieldId::Vertex ? vertex_fields
                     : id.prim == PyFieldId::Face   ? face_fields
                                                    : halfedge_fields;
  const int i = id_to_field.get_default(id.id,-1);
  if (i < 0)
    throw KeyError("no such mesh field");
  const UntypedArray& field = fields[i];

  if (id.type && &(field.type()) != id.type)
    throw ValueError(format("Type mismatch: id: %s, field: %s", field.type().name(), id.type->name()));

  #define CASE(...) \
    if (field.type() == typeid(__VA_ARGS__)) \
      return to_python_ref(field.get<__VA_ARGS__>());
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
  throw TypeError(format("Can't handle python conversion of fields of type %s", id.type->name()));
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

static string id_error(const TriangleTopology& mesh, const VertexId x) {
  return x.id==invalid_id              ? "invalid vertex id"
       : x.id==erased_id               ? "erased vertex id"
       : format(  !mesh.vertex_to_edge_.valid(x) ? "out of range vertex %d"
                : mesh.erased(x)                 ? "erased vertex %d"
                                                 : "internal error on vertex %d",x.id);
}
static string id_error(const TriangleTopology& mesh, const FaceId x) {
  return x.id==invalid_id              ? "invalid face id"
       : x.id==erased_id               ? "erased face id"
       : format(  !mesh.faces_.valid(x) ? "out of range face %d"
                : mesh.erased(x)        ? "erased face %d"
                                        : "internal error on face %d",x.id);
}
static string id_error(const TriangleTopology& mesh, const HalfedgeId x) {
  if (x.id==invalid_id) return "invalid halfedge id";
  if (x.id==erased_id) return "erased halfedge id";
    return format(  (x.id>=0 ? !mesh.faces_.valid(FaceId(x.id/3))
                             : !mesh.boundaries_.valid(-1-x.id)) ? "out of range interior halfedge %d"
                  : mesh.erased(x)                               ? "interior halfedge %d in erased face"
                                                                 : "internal error on halfedge %d",x.id);
}

// Safe versions of functions for python
#define MAKE_SAFE_2(ff,f,Id) \
  decltype((*(const TriangleTopology*)0).f(Id())) TriangleTopology::safe_##ff(Id x) const { \
    if (!valid(x)) \
      throw ValueError(format("TriangleTopology::" #ff ": %s",id_error(*this,x))); \
    return f(x); \
  }
#define MAKE_SAFE(f,Id) MAKE_SAFE_2(f,f,Id)
MAKE_SAFE(halfedge,VertexId)
MAKE_SAFE(prev,    HalfedgeId)
MAKE_SAFE(next,    HalfedgeId)
MAKE_SAFE(reverse,    HalfedgeId)
MAKE_SAFE(src,     HalfedgeId)
MAKE_SAFE(dst,     HalfedgeId)
MAKE_SAFE(face,    HalfedgeId)
MAKE_SAFE(left,    HalfedgeId)
MAKE_SAFE(right,   HalfedgeId)
MAKE_SAFE_2(face_vertices,    vertices, FaceId)
MAKE_SAFE_2(face_halfedges,   halfedges,FaceId)
MAKE_SAFE_2(halfedge_vertices,vertices, HalfedgeId)
MAKE_SAFE_2(face_faces,       faces,    FaceId)
MAKE_SAFE_2(halfedge_faces,   faces,    HalfedgeId)
MAKE_SAFE(outgoing,VertexId)
MAKE_SAFE(incoming,VertexId)

HalfedgeId TriangleTopology::safe_halfedge_between(VertexId v0, VertexId v1) const {
  const bool b0 = !valid(v0),
             b1 = !valid(v1);
  if (b0 || b1)
    throw ValueError(format("TriangleTopology::halfedge_between: %s vertex error: %s",
      b0 ? "first" : "second",id_error(*this,b0 ? v0 : v1)));
  return halfedge(v0,v1);
}

Tuple<Ref<SegmentSoup>,Array<HalfedgeId>> TriangleTopology::edge_soup() const {
  Array<Vector<int,2>> edges;
  Array<HalfedgeId> indices;

  for (auto i : halfedges()) {
    // only store the smaller of the two indices
    HalfedgeId r = reverse(i);
    if (!is_boundary(r) && r < i)
      continue;

    edges.append(vec(src(i).idx(), dst(i).idx()));
    indices.append(i);
  }

  return tuple(new_<SegmentSoup>(edges,allocated_vertices()), indices);
}

Tuple<Ref<TriangleSoup>,Array<FaceId>> TriangleTopology::face_soup() const {
  Array<Vector<int,3>> facets;
  Array<FaceId> indices;

  for (auto i : faces()) {
    auto verts = vertices(i);
    facets.append(vec(verts.x.idx(), verts.y.idx(), verts.z.idx()));
    indices.append(i);
  }

  return tuple(new_<TriangleSoup>(facets, allocated_vertices()), indices);
}

real TriangleTopology::area(RawField<const TV3,VertexId> X, const FaceId f) const {
  return triangle(X,f).area();
}

TV3 TriangleTopology::normal(RawField<const TV3,VertexId> X, const FaceId f) const {
  const auto v = vertices(f);
  return normal_cross(X[v.y]-X[v.x],X[v.z]-X[v.x]);
}

TV3 TriangleTopology::normal(RawField<const TV3,VertexId> X, const VertexId v) const {
  // Area weight normals
  TV3 n;
  for (const auto e : outgoing(v)) {
    const auto f = face(e);
    if (f.valid())
      n += angle_at(X,e)*normal(X,f);
  }

  // We're nondegenerate, return immediately
  const T nn = sqr_magnitude(n);
  if (nn)
    return n/sqrt(nn);

  // If we're degenerate, return the normal of the least degenerate face
  T max_angle = -inf;
  for (const auto e : outgoing(v)) {
    const auto f = face(e);
    if (f.valid()) {
      const T angle = angle_at(X,e);
      if (!(max_angle >= angle)) { // Invert comparison so that nans mean yes
        max_angle = angle;
        n = normal(X,f);
      }
    }
  }
  return n;
}

T TriangleTopology::dihedral(RawField<const TV3,VertexId> X, const HalfedgeId e) const {
  const auto t0 = triangle(X,face(e)),
             t1 = triangle(X,face(reverse(e)));
  const auto d = t1.center()-t0.center();
  const T abs_theta = acos(min(1.,max(-1.,dot(t0.n,t1.n))));
  return copysign(abs_theta,dot(t1.n-t0.n,d));
}

T TriangleTopology::cos_dihedral(RawField<const TV3,VertexId> X, const HalfedgeId e) const {
  const auto x1 = X[src(e)],
             x2 = X[dst(e)],
             x0 = X[dst(next(e))],
             x3 = X[dst(next(reverse(e)))],
             n0 = cross(x2-x1,x0-x1),
             n1 = cross(x3-x1,x0-x1);
  // Assume really bad dihedrals for degenerate elements
  const T denom = sqrt(sqr_magnitude(n0)*sqr_magnitude(n1));
  return denom ? dot(n0,n1)/denom : 1;
}

#define PER_DIMENSION(TV) \
  T TriangleTopology::angle_at(RawField<const TV,VertexId> X, const HalfedgeId e) const { \
    const auto x0 = X[src(e)], \
               x1 = X[dst(e)], \
               x2 = X[src(prev(e))]; \
    return angle_between(x1-x0,x2-x0); \
  } \
  Triangle<TV> TriangleTopology::triangle(RawField<const TV,VertexId> X, const FaceId f) const { \
    const auto v = vertices(f); \
    return Triangle<TV>(X[v.x],X[v.y],X[v.z]); \
  } \
  Segment<TV> TriangleTopology::segment(RawField<const TV,VertexId> X, const HalfedgeId e) const { \
    const auto v = vertices(e); \
    return Segment<TV>(X[v.x],X[v.y]); \
  } \
  real TriangleTopology::edge_length(RawField<const TV,VertexId> X, const HalfedgeId e) const{ \
    const auto x0 = X[src(e)], \
               x1 = X[dst(e)]; \
      return (x0-x1).magnitude(); \
  } \
  Tuple<Ref<SimplexTree<TV,1>>,Array<HalfedgeId>> TriangleTopology::edge_tree(Field<const TV,VertexId> X, const int leaf_size) const { \
    const auto soup = edge_soup(); \
    return tuple(new_<SimplexTree<TV,1>>(soup.x,X.flat,leaf_size),soup.y); \
  } \
  Tuple<Ref<SimplexTree<TV,2>>,Array<FaceId>> TriangleTopology::face_tree(Field<const TV,VertexId> X, const int leaf_size) const { \
    const auto soup = face_soup(); \
    return tuple(new_<SimplexTree<TV,2>>(soup.x,X.flat,leaf_size),soup.y); \
  }
PER_DIMENSION(TV2)
PER_DIMENSION(TV3)

#ifdef GEODE_PYTHON
Ref<> TriangleTopology::edge_tree_py(Array<const T,2> X) const {
  if (X.n==2)      return to_python_ref(edge_tree(Field<const TV2,VertexId>(vector_view_own<2>(X.flat))));
  else if (X.n==3) return to_python_ref(edge_tree(Field<const TV3,VertexId>(vector_view_own<3>(X.flat))));
  throw ValueError(format("TriangleTopology::edge_tree: Expected 2D or 3D vectors, got shape %s",str(X.sizes())));
}

Ref<> TriangleTopology::face_tree_py(Array<const T,2> X) const {
  if (X.n==2)      return to_python_ref(face_tree(Field<const TV2,VertexId>(vector_view_own<2>(X.flat))));
  else if (X.n==3) return to_python_ref(face_tree(Field<const TV3,VertexId>(vector_view_own<3>(X.flat))));
  throw ValueError(format("TriangleTopology::face_tree: Expected 2D or 3D vectors, got shape %s",str(X.sizes())));
}
#endif

#define SAFE_ERASE(prim,Id) \
  void MutableTriangleTopology::safe_erase_##prim(Id x, bool erase_isolated) { \
    if (!valid(x)) \
      throw ValueError(format("MutableTriangleTopology::erase_" #prim ": %s",id_error(*this,x))); \
    erase(x); \
  }
SAFE_ERASE(face,FaceId)
SAFE_ERASE(vertex,VertexId)
SAFE_ERASE(halfedge,HalfedgeId)

}
using namespace geode;

#include <geode/python/pyrange.h>

#ifdef GEODE_PYTHON
namespace geode {
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIncoming>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyOutgoing>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<VertexId>>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<FaceId>>);
template<> GEODE_DEFINE_TYPE(PyRange<TriangleTopologyIter<HalfedgeId>>);
}
#endif

void wrap_corner_mesh() {
  #define SAFE_METHOD(name) GEODE_METHOD_2(#name,safe_##name)
  {
    typedef TriangleTopology Self;
    Class<Self>("TriangleTopology")
      .GEODE_INIT(const TriangleSoup&)
      .GEODE_METHOD(copy)
      .GEODE_METHOD(mutate)
      .GEODE_GET(n_vertices)
      .GEODE_GET(n_boundary_edges)
      .GEODE_GET(n_edges)
      .GEODE_GET(n_faces)
      .GEODE_GET(allocated_vertices)
      .GEODE_GET(allocated_faces)
      .GEODE_GET(allocated_halfedges)
      .GEODE_GET(chi)
      .SAFE_METHOD(halfedge)
      .SAFE_METHOD(prev)
      .SAFE_METHOD(next)
      .SAFE_METHOD(src)
      .SAFE_METHOD(dst)
      .SAFE_METHOD(face)
      .SAFE_METHOD(left)
      .SAFE_METHOD(right)
      .SAFE_METHOD(face_vertices)
      .SAFE_METHOD(face_halfedges)
      .SAFE_METHOD(halfedge_vertices)
      .SAFE_METHOD(face_faces)
      .SAFE_METHOD(halfedge_faces)
      .SAFE_METHOD(outgoing)
      .SAFE_METHOD(incoming)
      .SAFE_METHOD(reverse)
      .GEODE_METHOD(vertex_one_ring)
      .GEODE_METHOD(incident_faces)
      .GEODE_METHOD(face_soup)
      .GEODE_OVERLOADED_METHOD_2(HalfedgeId(Self::*)(VertexId, VertexId)const, "halfedge_between", halfedge)
      .GEODE_METHOD(common_halfedge)
      .GEODE_OVERLOADED_METHOD_2(VertexId(Self::*)(FaceId, FaceId)const, "common_vertex_between_faces", common_vertex)
      .GEODE_OVERLOADED_METHOD_2(VertexId(Self::*)(FaceId, HalfedgeId)const, "common_vertex_between_face_and_halfedge", common_vertex)
      .GEODE_OVERLOADED_METHOD_2(HalfedgeId(Self::*)(FaceId, VertexId)const, "opposite_halfedge", opposite)
      .GEODE_OVERLOADED_METHOD_2(bool(Self::*)(VertexId)const, "is_boundary_vertex", is_boundary)
      .GEODE_OVERLOADED_METHOD_2(bool(Self::*)(HalfedgeId)const, "is_boundary_halfedge", is_boundary)
      .GEODE_METHOD(elements)
      .GEODE_METHOD(degree)
      .GEODE_METHOD(surface_components)
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
      .GEODE_METHOD(area)
      .GEODE_OVERLOADED_METHOD_2(TV3(Self::*)(RawField<const TV3,VertexId> X, const FaceId f) const, "face_normal", normal)
      .GEODE_OVERLOADED_METHOD_2(TV3(Self::*)(RawField<const TV3,VertexId> X, const  VertexId v) const, "vertex_normal", normal)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<VertexId>>(Self::*)() const, vertices)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<FaceId>>(Self::*)() const, faces)
      .GEODE_OVERLOADED_METHOD(Range<TriangleTopologyIter<HalfedgeId>>(Self::*)() const, halfedges)
      .GEODE_OVERLOADED_METHOD_2(bool(Self::*)(VertexId) const, "vertex_valid", valid)

      .GEODE_OVERLOADED_METHOD_2(bool(Self::*)(FaceId) const, "face_valid", valid)
      .GEODE_OVERLOADED_METHOD_2(bool(Self::*)(HalfedgeId) const, "halfedge_valid", valid)
      .GEODE_METHOD(boundary_edges)
      .GEODE_METHOD(interior_halfedges)
      .GEODE_METHOD(is_garbage_collected)
#ifdef GEODE_PYTHON
      .GEODE_METHOD_2("edge_tree",edge_tree_py)
      .GEODE_METHOD_2("face_tree",face_tree_py)
#endif
      ;
  }
  {
    typedef MutableTriangleTopology Self;
    Class<Self>("MutableTriangleTopology")
      .GEODE_INIT()
      .GEODE_METHOD(copy)
      .GEODE_METHOD(add)
      .GEODE_METHOD(flip)
      .GEODE_METHOD(flipped)
      .GEODE_METHOD(add_vertex)
      .GEODE_METHOD(add_vertices)
      .GEODE_METHOD(add_face)
      .GEODE_METHOD(add_faces)
      .SAFE_METHOD(erase_face)
      .SAFE_METHOD(erase_vertex)
      .SAFE_METHOD(erase_halfedge)
      .GEODE_METHOD(split_nonmanifold_vertex)
      .GEODE_METHOD(split_nonmanifold_vertices)
      .GEODE_METHOD(split_along_edge)
      .GEODE_METHOD(is_collapse_safe)
      .GEODE_METHOD(collapse)
      .GEODE_OVERLOADED_METHOD_2(VertexId(Self::*)(HalfedgeId),"split_edge",split_edge)
      .GEODE_OVERLOADED_METHOD_2(void(Self::*)(HalfedgeId,VertexId),"split_edge_with_vertex",split_edge)
      .GEODE_METHOD(erase_isolated_vertices)
      .GEODE_METHOD(collect_garbage)
      .GEODE_METHOD(collect_boundary_garbage)
      #ifdef GEODE_PYTHON
      .GEODE_METHOD_2("add_vertex_field",add_vertex_field_py)
      .GEODE_METHOD_2("add_face_field",add_face_field_py)
      .GEODE_METHOD_2("add_halfedge_field",add_halfedge_field_py)
      .GEODE_METHOD_2("has_field",has_field_py)
      .GEODE_METHOD_2("has_vertex_field",has_vertex_field_py)
      .GEODE_METHOD_2("has_face_field",has_face_field_py)
      .GEODE_METHOD_2("has_halfedge_field",has_halfedge_field_py)
      .GEODE_METHOD_2("remove_field",remove_field_py)
      .GEODE_METHOD_2("remove_vertex_field",remove_vertex_field_py)
      .GEODE_METHOD_2("remove_face_field",remove_face_field_py)
      .GEODE_METHOD_2("remove_halfedge_field",remove_halfedge_field_py)
      .GEODE_METHOD_2("field",field_py)
      .GEODE_METHOD_2("vertex_field",vertex_field_py)
      .GEODE_METHOD_2("face_field",face_field_py)
      .GEODE_METHOD_2("halfedge_field",halfedge_field_py)
      #endif
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
