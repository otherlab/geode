// Randomized incremental Delaunay using simulation of simplicity

#include <geode/exact/delaunay.h>
#include <geode/exact/predicates.h>
#include <geode/exact/quantize.h>
#include <geode/exact/scope.h>
#include <geode/array/amap.h>
#include <geode/array/RawField.h>
#include <geode/math/integer_log.h>
#include <geode/python/wrap.h>
#include <geode/random/permute.h>
#include <geode/random/Random.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/curry.h>
#include <geode/utility/interrupts.h>
#include <geode/utility/Log.h>
namespace geode {

using Log::cout;
using std::endl;
typedef Vector<real,2> TV;
typedef Vector<Quantized,2> EV;
using exact::Perturbed2;
const auto bound = exact::bound;

// Whether to run extremely expensive diagnostics
static const bool self_check = false;

// For interface simplicity, we use a single fixed random number as the seed.
// This is safe unless the points are chosen maliciously (and then it's still
// quite hard since threefry is fairly strong for a noncryptographic PRNG).
// The key itself came from /dev/random. :)
static const uint128_t key = uint128_t(9975794406056834021u) +(uint128_t(920519151720167868u)<<64);

// For details and analysis of the algorithm, see
//
//   Leonidas Guibas, Donald Knuth, Micha Sharir, "Randomized incremental construction of Delaunay and Voronoi diagrams".
//
// We define a BSP DAG based on the history of the algorithm.  Internal nodes correspond to halfspace tests against edges
// (which may no longer exist), and leaf nodes are currently existing triangles.  It's a DAG rather than a tree because
// we need to be able to incrementally replace leaf nodes with new subtrees, and it's impossible to ensure a unique path
// from root to leaf (both face splits and edge flips break this property).  Moreover, in the edge flip case two leaf nodes
// suddenly merge into one interior node.
//
// In addition to the BSP structure, we store a mapping from faces to all references in the BSP DAG.  Happily, there
// are at most two of these, which is easy to see by induction:
// 1. Face split: we destroy one triangle and create three new ones with reference counts 1,2,1.
// 2. Edge flip: we destroy two triangles and create two new ones, each with a single reference.
// Thus, it suffices to store two ints per face, one of which is possibly empty (-1).
//
// Randomized incremental Delaunay by itself has terrible cache properties, since the mesh is accessed in nearly purely
// random order.  Therefore, we use the elegant partial randomization idea of
//
//   Nina Amenta, Sunghee Choi, Gunter Rote, "Incremental Constructions con BRIO".
//
// which randomly assigns vertices into bins of exponentially increasing size, then applies a spatial sort within each bin.
// The random choice of bin is sufficient to ensure an O(n log n) worst case time, and the spatial sort produces a relatively
// cache coherent access pattern.
//
// For constrained Delaunay, we first perform randomized incremental insertion and then enforce constraints
// in random order using Shewchunk and Brown's algorithm:
//
//   Jonathan Richard Shewchuk and Brielin C. Brown. "Fast segment insertion and incremental construction of
//   constrained delaunay triangulations." In Proceedings of the 29th annual symposium on Symposium on
//   computational geometry, pp. 299-308. ACM, 2013.

namespace {
struct Node {
  Vector<VertexId,2> test; // Vertices defining the halfspace test
  Vector<int,2> children; // Right and left children: either n >= 0 for an internal node or ~f.id < 0 for a face.

  Node() {}
  Node(Vector<VertexId,2> test, int right, int left)
    : test(test), children(right,left) {}
};
}

DelaunayConstraintConflict::DelaunayConstraintConflict(const Vector<int,2> new_e0, const Vector<int,2> new_e1)
 : Base(format("delaunay: Constraints (%d,%d) and (%d,%d) intersect", new_e0.x, new_e0.y, new_e1.x, new_e1.y))
 , e0(new_e0)
 , e1(new_e1)
{ }

DelaunayConstraintConflict::~DelaunayConstraintConflict() throw () {}


// When a BSP leaf face is replaced by a node, update the links to point to the new node.
static inline void set_links(RawArray<Node> bsp, const Vector<int,2> links, const int node) {
  bsp[links.x>>1].children[links.x&1] = node; // The first link is always valid
  if (links.y>=0) // The second might not be
    bsp[links.y>>1].children[links.y&1] = node;
}

static inline bool is_sentinel(const Perturbed2& x) {
  return abs(x.value().x)==bound;
}

// Different sentinels will be placed at different orders of infinity, with earlier indices further away.
// These macros are used in the sorting networks below to move large sentinels last
#define SENTINEL_KEY(i) (is_sentinel(x##i)?-x##i.seed():numeric_limits<int>::min())
#define COMPARATOR(i,j) if (SENTINEL_KEY(i)>SENTINEL_KEY(j)) { swap(x##i,x##j); parity ^= 1; }

GEODE_COLD GEODE_PURE static inline bool triangle_oriented_sentinels(Perturbed2 x0, Perturbed2 x1, Perturbed2 x2) {
  // Move large sentinels last
  bool parity = 0;
  COMPARATOR(0,1)
  COMPARATOR(1,2)
  COMPARATOR(0,1)
  // Triangle orientation tests reduce to segment vs. direction and direction vs. direction when infinities are involved
  return parity ^ (!is_sentinel(x1) ? segment_to_direction_oriented(x0,x1,x2)   // one sentinel
                                    :           directions_oriented(   x1,x2)); // two or three sentinels
}

GEODE_ALWAYS_INLINE static inline bool triangle_oriented(const RawField<const Perturbed2,VertexId> X, VertexId v0, VertexId v1, VertexId v2) {
  const auto x0 = X[v0],
             x1 = X[v1],
             x2 = X[v2];
  // If we have a nonsentinel triangle, use a normal orientation test
  if (maxabs(x0.value().x,x1.value().x,x2.value().x)!=bound)
    return triangle_oriented(x0,x1,x2);
  // Fall back to sentinel case analysis
  return triangle_oriented_sentinels(x0,x1,x2);
}

// Test whether an edge containing sentinels is Delaunay
GEODE_COLD GEODE_PURE static inline bool is_delaunay_sentinels(Perturbed2 x0, Perturbed2 x1, Perturbed2 x2, Perturbed2 x3) {
  // Unfortunately, the sentinels need to be at infinity for purposes of Delaunay testing, and our SOS predicates
  // don't support infinities.  Therefore, we need some case analysis.  First, we move all the sentinels to the end,
  // sorted in decreasing order of index.  Different sentinels will be placed at different orders of infinity,
  // with earlier indices further away, so our order will place larger infinities last.
  bool parity = 0;
  COMPARATOR(0,1)
  COMPARATOR(2,3)
  COMPARATOR(0,2)
  COMPARATOR(1,3)
  COMPARATOR(1,2)
  if (!is_sentinel(x2)) // One infinity: A finite circle contains infinity iff it is inside out, so we reduce to an orientation test
    return parity^triangle_oriented(x0,x1,x2);
  else if (!is_sentinel(x1)) // Two infinities: also an orientation test, but with the last point at infinity
    return parity^segment_to_direction_oriented(x0,x1,x2);
  else // Three infinities: the finite point no longer matters.
    return parity^directions_oriented(x1,x2);
}

// Test whether an edge is Delaunay
GEODE_ALWAYS_INLINE static inline bool is_delaunay(const TriangleTopology& mesh, RawField<const Perturbed2,VertexId> X, const HalfedgeId edge) {
  // Boundary edges belong to the sentinel quad and are always Delaunay.
  const auto rev = mesh.reverse(edge);
  if (mesh.is_boundary(rev))
    return true;
  // Grab vertices in counterclockwise order
  const auto v0 = mesh.src(edge),
             v1 = mesh.src(mesh.prev(rev)),
             v2 = mesh.dst(edge),
             v3 = mesh.src(mesh.prev(edge));
  const auto x0 = X[v0],
             x1 = X[v1],
             x2 = X[v2],
             x3 = X[v3];
  // If we have a nonsentinel interior edge, perform a normal incircle test
  if (maxabs(x0.value().x,x1.value().x,x2.value().x,x3.value().x)!=bound)
    return !incircle(x0,x1,x2,x3);
  // Fall back to sentinel case analysis
  return is_delaunay_sentinels(x0,x1,x2,x3);
}

static inline FaceId bsp_search(RawArray<const Node> bsp, RawField<const Perturbed2,VertexId> X, const VertexId v) {
  if (!bsp.size())
    return FaceId(0);
  int node = 0;
  do {
    const Node& n = bsp[node];
    node = n.children[triangle_oriented(X,n.test.x,n.test.y,v)];
  } while (node >= 0);
  return FaceId(~node);
}

GEODE_UNUSED static void check_bsp(const TriangleTopology& mesh, RawArray<const Node> bsp, RawField<const Vector<int,2>,FaceId> face_to_bsp, RawField<const Perturbed2,VertexId> X_) {
  if (self_check) {
    cout << "bsp:\n";
    #define CHILD(c) format("%c%d",(c<0?'f':'n'),(c<0?~c:c))
    for (const int n : range(bsp.size()))
      cout << "  "<<n<<" : test "<<bsp[n].test<<", children "<<CHILD(bsp[n].children[0])<<" "<<CHILD(bsp[n].children[1])<<endl;
    cout << "tris = "<<mesh.elements()<<endl;
    cout << "X = "<<X_.flat<<endl;
  }
  for (const int n : range(bsp.size()))
    for (const int i : range(2))
      if (bsp[n].children[i]<0)
        GEODE_ASSERT(face_to_bsp[FaceId(~bsp[n].children[i])].contains(2*n+i));
  auto X = X_.copy();
  for (const auto f : mesh.faces()) {
    int i0,i1;
    face_to_bsp[f].get(i0,i1);
    if (bsp.size())
      GEODE_ASSERT(bsp[i0/2].children[i0&1]==~f.id);
    if (i1>=0)
      GEODE_ASSERT(bsp[i1/2].children[i1&1]==~f.id);
    const auto v = mesh.vertices(f);
    if (!is_sentinel(X[v.x]) && !is_sentinel(X[v.y]) && !is_sentinel(X[v.z])) {
      const auto center = X.append(Perturbed2(X.size(),(X[v.x].value()+X[v.y].value()+X[v.z].value())/3));
      const auto found = bsp_search(bsp,X,center);
      if (found!=f) {
        cout << "bsp search failed: f "<<f<<", v "<<v<<", found "<<found<<endl;
        GEODE_ASSERT(false);
      }
      X.flat.pop();
    }
  }
}

GEODE_COLD static void assert_delaunay(const char* prefix,
                                       const TriangleTopology& mesh, RawField<const Perturbed2,VertexId> X,
                                       const Hashtable<Vector<VertexId,2>>& constrained=Tuple<>(),
                                       const bool oriented_only=false,
                                       const bool check_boundary=true) {
  // Verify that all faces are correctly oriented
  for (const auto f : mesh.faces()) {
    const auto v = mesh.vertices(f);
    GEODE_ASSERT(triangle_oriented(X,v.x,v.y,v.z));
  }
  if (oriented_only)
    return;
  // Verify that all internal edges are Delaunay
  if (!constrained.size()) {
    for (const auto e : mesh.interior_halfedges())
      if (!mesh.is_boundary(mesh.reverse(e)) && mesh.src(e)<mesh.dst(e))
        if (!is_delaunay(mesh,X,e))
          throw RuntimeError(format("%snon delaunay edge: e%d, v%d v%d",prefix,e.id,mesh.src(e).id,mesh.dst(e).id));
  } else {
    for (const auto v : constrained)
      if (!mesh.halfedge(v.x,v.y).valid())
        throw RuntimeError(format("%smissing constraint edge: v%d v%d",prefix,v.x.id,v.y.id));
    for (const auto e : mesh.interior_halfedges()) {
      const auto s = mesh.src(e), d = mesh.dst(e);
      if (!mesh.is_boundary(mesh.reverse(e)) && s<d && !constrained.contains(vec(s,d)))
        if (!is_delaunay(mesh,X,e))
          throw RuntimeError(format("%snon delaunay edge: e%d, v%d v%d",prefix,e.id,mesh.src(e).id,mesh.dst(e).id));
    }
  }
  // Verify that all boundary vertices are convex
  if (check_boundary)
    for (const auto e : mesh.boundary_edges()) {
      const auto v0 = mesh.src(mesh.prev(e)),
                 v1 = mesh.src(e),
                 v2 = mesh.dst(e);
      GEODE_ASSERT(triangle_oriented(X,v2,v1,v0));
    }
}

GEODE_COLD static void assert_delaunay(const char* prefix,
                                       const TriangleTopology& mesh, RawField<const EV,VertexId> X,
                                       const Hashtable<Vector<VertexId,2>>& constrained=Tuple<>(),
                                       const bool oriented_only=false,
                                       const bool check_boundary=true) {
  Field<Perturbed2,VertexId> Xp(X.size(),uninit);
  for (const int i : range(X.size()))
    Xp.flat[i] = Perturbed2(i,X.flat[i]);
  assert_delaunay(prefix,mesh,Xp,constrained,oriented_only,check_boundary);
}


// This routine assumes the sentinel points have already been added, and processes points in order
GEODE_NEVER_INLINE static Ref<MutableTriangleTopology> deterministic_exact_delaunay(RawField<const Perturbed2,VertexId> X, const bool validate) {

  const int n = X.size()-3;
  const auto mesh = new_<MutableTriangleTopology>();
  IntervalScope scope;

  // Initialize the mesh to a Delaunay triangle containing the sentinels at infinity.
  mesh->add_vertices(n+3);
  mesh->add_face(vec(VertexId(n+0),VertexId(n+1),VertexId(n+2)));
  if (self_check) {
    mesh->assert_consistent();
    assert_delaunay("self check 0: ",mesh,X);
  }

  // The randomized incremental construction algorithm uses the history of the triangles
  // as the acceleration structure.  Specifically, we maintain a BSP tree where the nodes
  // are edge tests (are we only the left or right of an edge) and the leaves are triangles.
  // There are two operations that modify this tree:
  //
  // 1. Split: A face is split into three by the insertion of an interior vertex.
  // 2. Flip: An edge is flipped, turning two triangles into two different triangles.
  //
  // The three starts out empty, since one triangle needs zero tests.
  Array<Node> bsp; // All BSP nodes including leaves
  bsp.preallocate(3*n); // The minimum number of possible BSP nodes
  Field<Vector<int,2>,FaceId> face_to_bsp; // Map from FaceId to up to two BSP leaf points (2*node+(right?0:1))
  face_to_bsp.flat.preallocate(2*n+1); // The exact maximum number of faces
  face_to_bsp.flat.append_assuming_enough_space(vec(0,-1)); // By the time we call set_links, node 0 will be valid
  if (self_check)
    check_bsp(*mesh,bsp,face_to_bsp,X);

  // Allocate a stack to simulate recursion when flipping non-Delaunay edges.
  // Invariant: if edge is on the stack, the other edges of face(edge) are Delaunay.
  // Since halfedge ids change during edge flips in a corner mesh, we store half edges as directed vertex pairs.
  Array<Tuple<HalfedgeId,Vector<VertexId,2>>> stack;
  stack.preallocate(8);

  // Insert all vertices into the mesh in random order, maintaining the Delaunay property
  for (const auto i : range(n)) {
    const VertexId v(i);
    check_interrupts();

    // Search through the BSP tree to find the containing triangle
    const auto f0 = bsp_search(bsp,X,v);
    const auto vs = mesh->vertices(f0);

    // Split the face by inserting the new vertex and update the BSP tree accordingly.
    mesh->split_face(f0,v);
    const auto e0 = mesh->halfedge(v),
               e1 = mesh->left(e0),
               e2 = mesh->right(e0);
    assert(mesh->dst(e0)==vs.x);
    const auto f1 = mesh->face(e1),
               f2 = mesh->face(e2);
    const int base = bsp.extend(3,uninit);
    set_links(bsp,face_to_bsp[f0],base);
    bsp[base+0] = Node(vec(v,vs.x),base+2,base+1);
    bsp[base+1] = Node(vec(v,vs.y),~f0.id,~f1.id);
    bsp[base+2] = Node(vec(v,vs.z),~f1.id,~f2.id);
    face_to_bsp[f0] = vec(2*(base+1)+0,-1);
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+1)+1,2*(base+2)+0));
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+2)+1,-1));
    if (self_check) {
      mesh->assert_consistent();
      check_bsp(*mesh,bsp,face_to_bsp,X);
    }

    // Fix all non-Delaunay edges
    stack.copy(vec(tuple(mesh->next(e0),vec(vs.x,vs.y)),
                   tuple(mesh->next(e1),vec(vs.y,vs.z)),
                   tuple(mesh->next(e2),vec(vs.z,vs.x))));
    if (self_check)
      assert_delaunay("self check 1: ",mesh,X,Tuple<>(),true);
    while (stack.size()) {
      const auto evs = stack.pop();
      auto e = mesh->vertices(evs.x)==evs.y ? evs.x : mesh->halfedge(evs.y.x,evs.y.y);
      if (e.valid() && !is_delaunay(*mesh,X,e)) {
        // Our mesh is linearly embedded in the plane, so edge flips are always safe
        assert(mesh->is_flip_safe(e));
        e = mesh->unsafe_flip_edge(e);
        GEODE_ASSERT(is_delaunay(*mesh,X,e));
        // Update the BSP tree for the triangle flip
        const auto f0 = mesh->face(e),
                   f1 = mesh->face(mesh->reverse(e));
        const int node = bsp.append(Node(mesh->vertices(e),~f1.id,~f0.id));
        set_links(bsp,face_to_bsp[f0],node);
        set_links(bsp,face_to_bsp[f1],node);
        face_to_bsp[f0] = vec(2*node+1,-1);
        face_to_bsp[f1] = vec(2*node+0,-1);
        if (self_check) {
          mesh->assert_consistent();
          check_bsp(*mesh,bsp,face_to_bsp,X);
        }
        // Recurse to successor edges to e
        const auto e0 = mesh->next(e),
                   e1 = mesh->prev(mesh->reverse(e));
        stack.extend(vec(tuple(e0,mesh->vertices(e0)),
                         tuple(e1,mesh->vertices(e1))));
        if (self_check)
          assert_delaunay("self check 2: ",mesh,X,Tuple<>(),true);
      }
    }
    if (self_check) {
      mesh->assert_consistent();
      assert_delaunay("self check 3: ",mesh,X);
    }
  }

  // Remove sentinels
  for (int i=0;i<3;i++)
    mesh->erase_last_vertex_with_reordering();

  // If desired, check that the final mesh is Delaunay
  if (validate)
    assert_delaunay("delaunay validate: ",mesh,X);

  // Return the mesh with the sentinels removed
  return mesh;
}

// InsertVertex in the paper
static void insert_cavity_vertex_helper(MutableTriangleTopology& mesh, RawField<const Perturbed2,VertexId> X,
                                        RawField<bool,VertexId> marked, const HalfedgeId vw) {
  // If wv is a boundary edge, or we're already Delaunay and properly oriented, we're done
  const auto wv = mesh.reverse(vw);
  if (mesh.is_boundary(wv))
    return;
  const auto u = mesh.opposite(vw),
             v = mesh.src(vw),
             w = mesh.dst(vw),
             x = mesh.opposite(wv);
  const auto Xu = X[u],
             Xv = X[v],
             Xw = X[w],
             Xx = X[x];
  const bool in = incircle(Xu,Xv,Xw,Xx);
  if (!in && triangle_oriented(Xu,Xv,Xw))
    return;

  // Flip edge and recurse
  const auto xu = mesh.flip_edge(wv);
  assert(mesh.vertices(xu)==vec(x,u));
  const auto vx = mesh.prev(xu),
             xw = mesh.next(mesh.reverse(xu)); // Grab this now before the recursive call changes uvx
  insert_cavity_vertex_helper(mesh,X,marked,vx),
  insert_cavity_vertex_helper(mesh,X,marked,xw);
  if (!in)
    marked[u] = marked[v] = marked[w] = marked[x] = true;
}
static void insert_cavity_vertex(MutableTriangleTopology& mesh, RawField<const Perturbed2,VertexId> X,
                                 RawField<bool,VertexId> marked,
                                 const VertexId u, const VertexId v, const VertexId w) {
#ifndef NDEBUG
  {
    const auto vw = mesh.halfedge(v);
    assert(mesh.is_boundary(vw) && mesh.dst(vw)==w);
  }
#endif
  mesh.add_face(vec(u,v,w));
  const auto vw = mesh.prev(mesh.reverse(mesh.halfedge(u)));
  assert(mesh.vertices(vw)==vec(v,w));
  insert_cavity_vertex_helper(mesh,X,marked,vw);
}

// Allow unit tests to count how many times chew_fan is called.  Finding a Chew fan case
// is quite difficult, so it's important to verify that we've hit them.  Since chew_fan
// is called incredibly rarely, the slight slowdown is irrelevant.
static int chew_fan_count_ = 0;
static int chew_fan_count() {
  return chew_fan_count_;
}

// Delaunay retriangulate a triangle fan
static void chew_fan(MutableTriangleTopology& parent_mesh, RawField<const Perturbed2,VertexId> X,
                     const VertexId u, RawArray<HalfedgeId> fan, Random& random) {
  chew_fan_count_ += 1;
#ifndef NDEBUG
  for (const auto e : fan)
    assert(parent_mesh.opposite(e)==u);
  for (int i=0;i<fan.size()-1;i++)
    GEODE_ASSERT(parent_mesh.src(fan[i])==parent_mesh.dst(fan[i+1]));
#endif
  const int n = fan.size();
  if (n < 2)
    return;
  chew_fan_count_ += 1024*n;

  // Collect vertices
  const Field<VertexId,VertexId> vertices(n+2,uninit);
  vertices.flat[0] = u;
  vertices.flat[1] = parent_mesh.src(fan[n-1]);
  for (int i=0;i<n;i++)
    vertices.flat[i+2] = parent_mesh.dst(fan[n-1-i]);

  // Delete original vertices
  for (const auto e : fan)
    parent_mesh.erase(parent_mesh.face(e));

  // Make the vertices into a doubly linked list
  const Field<VertexId,VertexId> prev(n+2,uninit),
                                 next(n+2,uninit);
  prev.flat[0].id = n+1;
  next.flat[n+1].id = 0;
  for (int i=0;i<n+1;i++) {
    prev.flat[i+1].id = i;
    next.flat[i].id = i+1;
  }

  // Randomly shuffle the vertices, then pulling elements off the linked list in reverse order of our final shuffle.
  const Array<VertexId> pi(n+2,uninit);
  for (int i=0;i<n+2;i++)
    pi[i].id = i;
  random.shuffle(pi);
  for (int i=n+1;i>=0;i--) {
    const auto j = pi[i];
    prev[next[j]] = prev[j];
    next[prev[j]] = next[j];
  }

  // Make a new singleton mesh
  const auto mesh = new_<MutableTriangleTopology>();
  mesh->add_vertices(n+2);
  small_sort(pi[0],pi[1],pi[2]);
  mesh->add_face(vec(pi[0],pi[1],pi[2]));

  // Insert remaining vertices
  Array<HalfedgeId> work;
  for (int i=3;i<n+2;i++) {
    const auto j = pi[i];
    const auto f = mesh->add_face(vec(j,next[j],prev[j]));
    work.append(mesh->reverse(mesh->opposite(f,j)));
    while (work.size()) {
      auto e = work.pop();
      if (   !mesh->is_boundary(e)
          && incircle(X[vertices[mesh->src(e)]],
                      X[vertices[mesh->dst(e)]],
                      X[vertices[mesh->opposite(e)]],
                      X[vertices[mesh->opposite(mesh->reverse(e))]])) {
        work.append(mesh->reverse(mesh->next(e)));
        work.append(mesh->reverse(mesh->prev(e)));
        e = mesh->unsafe_flip_edge(e);
      }
    }
  }

  // Copy triangles back to parent
  for (const auto f : mesh->faces()) {
    const auto vs = mesh->vertices(f);
    parent_mesh.add_face(vec(vertices[vs.x],vertices[vs.y],vertices[vs.z]));
  }
}

// Retriangulate a cavity formed when a constraint edge is inserted, following Shewchuck and Brown.
// The cavity is defined by a counterclockwise list of vertices v[0] to v[m-1] as in Shewchuck and Brown, Figure 5.
static void cavity_delaunay(MutableTriangleTopology& parent_mesh, RawField<const EV,VertexId> X,
                            RawArray<const VertexId> cavity, Random& random) {
  // Since the algorithm generates meshes which may be inconsistent with the outer mesh, and the cavity
  // array may have duplicate vertices, we use a temporary mesh and then copy the triangles over when done.
  // In the temporary, vertices are indexed consecutively from 0 to m-1.
  const int m = cavity.size();
  assert(m >= 3);
  const auto mesh = new_<MutableTriangleTopology>();
  Field<Perturbed2,VertexId> Xc(m,uninit);
  for (const int i : range(m))
    Xc.flat[i] = Perturbed2(cavity[i].id,X[cavity[i]]);
  mesh->add_vertices(m);
  const auto xs = Xc.flat[0],
             xe = Xc.flat[m-1];

  // Set up data structures for prev, next, pi in the paper
  const Field<VertexId,VertexId> prev(m,uninit),
                                 next(m,uninit);
  for (int i=0;i<m-1;i++) {
    next.flat[i] = VertexId(i+1);
    prev.flat[i+1] = VertexId(i);
  }
  const Array<VertexId> pi_(m-2,uninit);
  for (int i=1;i<=m-2;i++)
    pi_[i-1] = VertexId(i);
  #define PI(i) pi_[(i)-1]

  // Randomly shuffle [1,m-2], subject to vertices closer to xs-xe than both their neighbors occurring later
  for (int i=m-2;i>=2;i--) {
    int j;
    for (;;) {
      j = random.uniform<int>(0,i)+1;
      const auto pj = PI(j);
      if (!(   segment_directions_oriented(xe,xs,Xc[pj],Xc[prev[pj]])
            && segment_directions_oriented(xe,xs,Xc[pj],Xc[next[pj]])))
        break;
    }
    swap(PI(i),PI(j));
    // Remove PI(i) from the list
    const auto pi = PI(i);
    next[prev[pi]] = next[pi];
    prev[next[pi]] = prev[pi];
  }

  // Add the first triangle
  mesh->add_face(vec(VertexId(0),PI(1),VertexId(m-1)));

  // Add remaining triangles, flipping to ensure Delaunay
  const Field<bool,VertexId> marked(m);
  Array<HalfedgeId> fan;
  bool used_chew = false;
  for (int i=2;i<m-1;i++) {
    const auto pi = PI(i);
    insert_cavity_vertex(mesh,Xc,marked,pi,next[pi],prev[pi]);
    if (marked[pi]) {
      used_chew = true;
      marked[pi] = false;
      // Retriangulate the fans of triangles that have all three vertices marked
      auto e = mesh->reverse(mesh->halfedge(pi));
      auto v = mesh->src(e);
      bool mv = marked[v];
      marked[v] = false;
      fan.clear();
      do {
        const auto h = mesh->prev(e);
        e = mesh->reverse(mesh->next(e));
        v = mesh->src(e);
        const bool mv2 = marked[v];
        marked[v] = false;
        if (mv) {
          if (mv2)
            fan.append(h);
          if (!mv2 || mesh->is_boundary(e)) {
            chew_fan(mesh,Xc,pi,fan,random);
            fan.clear();
          }
        }
        mv = mv2;
      } while (!mesh->is_boundary(e));
    }
  }
  #undef PI

  // If we ran Chew's algorithm, validate the output.  I haven't tested this
  // case enough to be confident of its correctness.
  if (used_chew)
    assert_delaunay("Failure in extreme special case.  If this triggers, please email geode-dev@googlegroups.com: ",
                    mesh,Xc,Tuple<>(),false,false);

  // Copy triangles from temporary mesh to real mesh
  for (const auto f : mesh->faces()) {
    const auto v = mesh->vertices(f);
    parent_mesh.add_face(vec(cavity[v.x.id],cavity[v.y.id],cavity[v.z.id]));
  }
}

GEODE_NEVER_INLINE static void add_constraint_edges(MutableTriangleTopology& mesh, RawField<const EV,VertexId> X,
                                                    RawArray<const Vector<int,2>> edges, const bool validate) {
  if (!edges.size())
    return;
  IntervalScope scope;
  Hashtable<Vector<VertexId,2>> constrained;
  Array<VertexId> left_cavity, right_cavity; // List of vertices for both cavities
  const auto random = new_<Random>(key+7);
  for (int i=0;i<edges.size();i++) {
    // Randomly choose an edge to ensure optimal time complexity
    const auto edge = edges[int(random_permute(edges.size(),key+5,i))].sorted();
    auto v0 = VertexId(edge.x),
         v1 = VertexId(edge.y);
    const auto vs = vec(v0,v1);
    GEODE_ASSERT(mesh.valid(v0) && mesh.valid(v1));

    {
      // Check if the edge already exists in the triangulation.  To ensure optimal complexity,
      // we loop around both vertices interleaved so that our time is O(min(degree(v0),degree(v1))).
      const auto s0 = mesh.halfedge(v0),
                 s1 = mesh.halfedge(v1);
      {
        auto e0 = s0,
             e1 = s1;
        do {
          if (mesh.dst(e0)==v1 || mesh.dst(e1)==v0)
            goto success; // The edge already exists, so there's nothing to be done.
          e0 = mesh.left(e0);
          e1 = mesh.left(e1);
        } while (e0!=s0 && e1!=s1);
      }

      // Find a triangle touching v0 or v1 containing part of the v0-v1 segment.
      // As above, we loop around both vertices interleaved.
      auto e0 = s0;
      {
        auto e1 = s1;
        if (mesh.is_boundary(e0)) e0 = mesh.left(e0);
        if (mesh.is_boundary(e1)) e1 = mesh.left(e1);
        const auto x0 = Perturbed2(v0.id,X[v0]),
                   x1 = Perturbed2(v1.id,X[v1]);
        const auto e0d = mesh.dst(e0),
                   e1d = mesh.dst(e1);
        bool e0o = triangle_oriented(x0,Perturbed2(e0d.id,X[e0d]),x1),
             e1o = triangle_oriented(x1,Perturbed2(e1d.id,X[e1d]),x0);
        for (;;) { // No need to check for an end condition, since we're guaranteed to terminate
          const auto n0 = mesh.left(e0),
                     n1 = mesh.left(e1);
          const auto n0d = mesh.dst(n0),
                     n1d = mesh.dst(n1);
          const bool n0o = triangle_oriented(x0,Perturbed2(n0d.id,X[n0d]),x1),
                     n1o = triangle_oriented(x1,Perturbed2(n1d.id,X[n1d]),x0);
          if (e0o && !n0o)
            break;
          if (e1o && !n1o) {
            // Swap v0 with v1 and e0 with e1 so that our ray starts at v0
            swap(v0,v1);
            swap(e0,e1);
            break;
          }
          e0 = n0;
          e1 = n1;
          e0o = n0o;
          e1o = n1o;
        }
      }

      // If we only need to walk one step, the retriangulation is a single edge flip
      auto cut = mesh.reverse(mesh.next(e0));
      if (mesh.dst(mesh.next(cut))==v1) {
        if (constrained.contains(vec(mesh.src(cut),mesh.dst(cut)).sorted()))
          throw DelaunayConstraintConflict(vec(v0.id,v1.id),vec(mesh.src(cut).id,mesh.dst(cut).id));
        cut = mesh.flip_edge(cut);
        goto success;
      }

      // Walk from v0 to v1, collecting the two cavities.
      const auto x0 = Perturbed2(v0.id,X[v0]),
                 x1 = Perturbed2(v1.id,X[v1]);
      right_cavity.copy(vec(v0,mesh.dst(cut)));
      left_cavity .copy(vec(v0,mesh.src(cut)));
      mesh.erase(mesh.face(e0));
      for (;;) {
        if (constrained.contains(vec(mesh.src(cut),mesh.dst(cut)).sorted()))
          throw DelaunayConstraintConflict(vec(v0.id,v1.id),vec(mesh.src(cut).id,mesh.dst(cut).id));
        const auto n = mesh.reverse(mesh.next(cut)),
                   p = mesh.reverse(mesh.prev(cut));
        const auto v = mesh.src(n);
        mesh.erase(mesh.face(cut));
        if (v == v1) {
          left_cavity.append(v);
          right_cavity.append(v);
          break;
        } else if (triangle_oriented(x0,x1,Perturbed2(v.id,X[v]))) {
          left_cavity.append(v);
          cut = n;
        } else {
          right_cavity.append(v);
          cut = p;
        }
      }

      // Retriangulate both cavities
      left_cavity.reverse();
      cavity_delaunay(mesh,X,left_cavity,random),
      cavity_delaunay(mesh,X,right_cavity,random);
    }
    success:
    constrained.set(vs);
  }

  // If desired, check that the final mesh is constrained Delaunay
  if (validate)
    assert_delaunay("constrained delaunay validate: ",mesh,X,constrained);
}

template<int axis> static inline int spatial_partition(RawArray<Perturbed2> X, Random& random) {
  // We use exact arithmetic to perform the partition, which is important in case many points are coincident
  #define LESS(i,j) (i!=j && axis_less<axis>(X[i],X[j]))

  // We partition by picking three elements at random, and running partition based on the middle element.
  const int n = X.size();
  int i0 = random.uniform<int>(0,n),
      i1 = random.uniform<int>(0,n),
      i2 = random.uniform<int>(0,n);
  if (!LESS(i0,i1)) swap(i0,i1);
  if (!LESS(i1,i2)) swap(i1,i2);
  if (!LESS(i0,i1)) swap(i0,i1);
  const auto Xmid = X[i1];
  swap(X[i1],X.back());

  // Perform the partition.  We use the version of partition from Wikipedia: http://en.wikipedia.org/wiki/Quicksort#In-place_version
  int mid = 0;
  for (const int i : range(n-1))
    if (axis_less<axis>(X[i],Xmid))
      swap(X[i],X[mid++]);
  return mid;
}

static void spatial_sort(RawArray<Perturbed2> X, const int leaf_size, Random& random) {
  const int n = X.size();
  if (n<=leaf_size)
    return;

  // We determine the subdivision axis using inexact computation, which is okay since neither the result nor
  // the asymptotic worst case complexity depends upon any properties of the spatial_sort whatsoever.
  const Box<EV> box = bounding_box(X.project<EV,&Perturbed2::value_>());
  const int axis = box.sizes().argmax();

  // We use exact arithmetic to perform the partition, which is important in case many points are coincident
  const int mid = axis==0 ? spatial_partition<0>(X,random)
                          : spatial_partition<1>(X,random);

  // Recursely sort both halves
  spatial_sort(X.slice(0,mid),leaf_size,random);
  spatial_sort(X.slice(mid,n),leaf_size,random);
}

// Prepare a list of points for Delaunay triangulation: randomly assign into logarithmic bins, sort within bins, and add sentinels.
// For details, see Amenta et al., Incremental Constructions con BRIO.
static Array<Perturbed2> partially_sorted_shuffle(RawArray<const EV> Xin) {
  const int n = Xin.size();
  Array<Perturbed2> X(n+3,uninit);

  // Randomly assign input points into bins.  Bin k has 2**k = 1,2,4,8,... and starts at index 2**k-1 = 0,1,3,7,...
  // We fill points into bins as sequentially as possible to maximize cache coherence.
  const int bins = integer_log(n);
  Array<int> bin_counts(bins);
  for (int i=0;i<n;i++) {
    int j = (int)random_permute(n,key,i);
    const int bin = min(integer_log(j+1),bins-1);
    j = (1<<bin)-1+bin_counts[bin]++;
    X[j] = Perturbed2(i,Xin[i]);
  }

  // Spatially sort each bin down to clusters of size 64.
  const int leaf_size = 64;
  for (int bin=0;bin<bins;bin++) {
    const int start = (1<<bin)-1,
              end = bin==bins-1?n:start+(1<<bin);
    assert(bin_counts[bin]==end-start);
    spatial_sort(X.slice(start,end),leaf_size,new_<Random>(key+bin));
  }

  // Add 3 sentinel points at infinity
  X[n+0] = Perturbed2(n+0,EV(-bound,-bound));
  X[n+1] = Perturbed2(n+1,EV( bound, 0)    );
  X[n+2] = Perturbed2(n+2,EV(-bound, bound));

  return X;
}

Ref<TriangleTopology> exact_delaunay_points(RawArray<const EV> X, RawArray<const Vector<int,2>> edges,
                                            const bool validate) {
  const int n = X.size();
  GEODE_ASSERT(n>=3);

  // Quantize all input points, reorder, and add sentinels
  Field<const Perturbed2,VertexId> Xp(partially_sorted_shuffle(X));

  // Compute Delaunay triangulation
  const auto mesh = deterministic_exact_delaunay(Xp,validate);

  // Undo the vertex permutation
  mesh->permute_vertices(Xp.flat.slice(0,n).project<int,&Perturbed2::seed_>().copy());

  // Insert constraint edges in random order
  add_constraint_edges(mesh,RawField<const EV,VertexId>(X),edges,validate);

  // All done!
  return mesh;
}

Ref<TriangleTopology> delaunay_points(RawArray<const Vector<real,2>> X, RawArray<const Vector<int,2>> edges,
                                      const bool validate) {
  return exact_delaunay_points(amap(quantizer(bounding_box(X)),X).copy(),edges,validate);
}

// Greedily compute a set of nonintersecting edges in a point cloud for testing purposes
// Warning: Takes O(n^3) time.
static Array<Vector<int,2>> greedy_nonintersecting_edges(RawArray<const Vector<real,2>> X, const int limit) {
  const auto E = amap(quantizer(bounding_box(X)),X).copy();
  const int n = E.size();
  Array<Vector<int,2>> edges;
  IntervalScope scope;
  for (const int i : range(n*n)) {
    const int pi = int(random_permute(n*n,key+14,i));
    const int a = pi/n,
              b = pi-n*a;
    if (a < b) {
      const auto Ea = Perturbed2(a,E[a]),
                 Eb = Perturbed2(b,E[b]);
      for (const auto e : edges)
        if (!e.contains(a) && !e.contains(b) && segments_intersect(Ea,Eb,Perturbed2(e.x,E[e.x]),Perturbed2(e.y,E[e.y])))
          goto skip;
      edges.append(vec(a,b));
      if (edges.size()==limit || edges.size()==3*n-6)
        break;
      skip:;
    }
  }
  return edges;
}

}
using namespace geode;

void wrap_delaunay() {
  GEODE_FUNCTION_2(delaunay_points_py,delaunay_points)
  GEODE_FUNCTION(greedy_nonintersecting_edges)
  GEODE_FUNCTION(chew_fan_count)
}
