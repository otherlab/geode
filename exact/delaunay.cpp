// Randomized incremental Delaunay using simulation of simplicity

#include <other/core/exact/delaunay.h>
#include <other/core/exact/predicates.h>
#include <other/core/exact/quantize.h>
#include <other/core/exact/scope.h>
#include <other/core/array/amap.h>
#include <other/core/array/RawField.h>
#include <other/core/mesh/HalfedgeMesh.h>
#include <other/core/python/wrap.h>
#include <other/core/random/permute.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
namespace other {

using Log::cout;
using std::endl;
typedef Vector<real,2> TV;
typedef Vector<exact::Quantized,2> EV;
const auto bound = exact::bound;

// Whether to run extremely expensive diagnostics
static const bool self_check = false;

// For interface simplicity, we use a single fixed random number as the seed.
// This is safe unless the points are chosen maliciously (and then it's still
// quite hard since threefry is fairly strong for a noncryptographic PRNG).
// The key itself came from /dev/random. :)
static const uint128_t key = 9975794406056834021u+(uint128_t(920519151720167868u)<<64);

// For details and analysis of the algorithm, see
//   Leonidas Guibas, Donald Knuth, Micha Sharir, "Randomized incremental construction of Delaunay and Voronoi diagrams".

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

namespace {
struct Node {
  Vector<VertexId,2> test; // Vertices defining the halfspace test
  Vector<int,2> children; // Right and left children: either n >= 0 for an internal node or ~f.id < 0 for a face.

  Node() {}
  Node(Vector<VertexId,2> test, int right, int left)
    : test(test), children(right,left) {}
};
}

// When a BSP leaf face is replaced by a node, update the links to point to the new node.
static inline void set_links(RawArray<Node> bsp, const Vector<int,2> links, const int node) {
  bsp[links.x>>1].children[links.x&1] = node; // The first link is always valid
  if (links.y>=0) // The second might not be
    bsp[links.y>>1].children[links.y&1] = node;
}

static inline bool is_sentinel(const EV& x) {
  return abs(x.x)==bound;
}

// Different sentinels will be placed at different orders of infinity, with earlier indices further away.
// These macros are used in the sorting networks below to move large sentinels last
#define SENTINEL_KEY(i) (is_sentinel(x##i)?-v##i.id:numeric_limits<int>::min())
#define COMPARATOR(i,j) if (SENTINEL_KEY(i)>SENTINEL_KEY(j)) { swap(v##i,v##j); swap(x##i,x##j); parity ^= 1; }

OTHER_COLD OTHER_CONST static inline bool triangle_oriented_sentinels(VertexId v0, EV x0, VertexId v1, EV x1, VertexId v2, EV x2) {
  // Move large sentinels last
  bool parity = 0;
  COMPARATOR(0,1)
  COMPARATOR(1,2)
  COMPARATOR(0,1)
  // Triangle orientation tests reduce to segment vs. direction and direction vs. direction when infinities are involved
  return parity ^ (!is_sentinel(x1) ? segment_to_direction_oriented(v0.id,x0,v1.id,x1,v2.id,x2)   // one sentinel
                                    :           directions_oriented(         v1.id,x1,v2.id,x2)); // two or three sentinels
}

OTHER_ALWAYS_INLINE static inline bool triangle_oriented(const RawField<const EV,VertexId> X, VertexId v0, VertexId v1, VertexId v2) {
  const auto x0 = X[v0],
             x1 = X[v1],
             x2 = X[v2];
  // If we have a nonsentinel triangle, use a normal orientation test
  if (maxabs(x0.x,x1.x,x2.x)!=bound)
    return triangle_oriented(v0.id,x0,v1.id,x1,v2.id,x2);
  // Fall back to sentinel case analysis
  return triangle_oriented_sentinels(v0,x0,v1,x1,v2,x2);
}

// Test whether an edge containing sentinels is Delaunay
OTHER_COLD OTHER_CONST static inline bool is_delaunay_sentinels(VertexId v0, EV x0, VertexId v1, EV x1, VertexId v2, EV x2, VertexId v3, EV x3) {
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
    return parity^triangle_oriented(v0.id,x0,v1.id,x1,v2.id,x2);
  else if (!is_sentinel(x1)) // Two infinities: also an orientation test, but with the last point at infinity
    return parity^segment_to_direction_oriented(v0.id,x0,v1.id,x1,v2.id,x2);
  else // Three infinities: the finite point no longer matters.
    return parity^directions_oriented(v1.id,x1,v2.id,x2);
}

// Test whether an edge is Delaunay
OTHER_ALWAYS_INLINE static inline bool is_delaunay(const HalfedgeMesh& mesh, RawField<const EV,VertexId> X, const HalfedgeId edge) {
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
  if (maxabs(x0.x,x1.x,x2.x,x3.x)!=bound)
    return !incircle(v0.id,x0,v1.id,x1,v2.id,x2,v3.id,x3);
  // Fall back to sentinel case analysis
  return is_delaunay_sentinels(v0,x0,v1,x1,v2,x2,v3,x3);
}

static inline FaceId bsp_search(RawArray<const Node> bsp, RawField<const EV,VertexId> X, const VertexId v) {
  if (!bsp.size())
    return FaceId(0);
  int iters = 0;
  int node = 0;
  do {
    iters++;
    const Node& n = bsp[node];
    node = n.children[triangle_oriented(X,n.test.x,n.test.y,v)];
  } while (node >= 0);
  return FaceId(~node);
}

OTHER_UNUSED static void check_bsp(const HalfedgeMesh& mesh, RawArray<const Node> bsp, RawField<const Vector<int,2>,FaceId> face_to_bsp, RawField<const EV,VertexId> X_) {
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
        OTHER_ASSERT(face_to_bsp[FaceId(~bsp[n].children[i])].contains(2*n+i));
  auto X = X_.copy();
  for (const auto f : mesh.faces()) {
    int i0,i1;
    face_to_bsp[f].get(i0,i1);
    if (bsp.size())
      OTHER_ASSERT(bsp[i0/2].children[i0&1]==~f.id);
    if (i1>=0)
      OTHER_ASSERT(bsp[i1/2].children[i1&1]==~f.id);
    const auto v = mesh.vertices(f);
    if (!is_sentinel(X[v.x]) && !is_sentinel(X[v.y]) && !is_sentinel(X[v.z])) {
      const auto center = X.append((X[v.x]+X[v.y]+X[v.z])/3);
      if (bsp_search(bsp,X,center)!=f) {
        cout << "bsp search failed: f "<<f<<", v "<<v<<endl;
        OTHER_ASSERT(false);
      }
      X.flat.pop();
    }
  }
}

OTHER_COLD static void assert_delaunay(const HalfedgeMesh& mesh, RawField<const EV,VertexId> X, const bool oriented_only=false) {
  // Verify that all faces are correctly oriented
  for (const auto f : mesh.faces()) {
    const auto v = mesh.vertices(f);
    OTHER_ASSERT(triangle_oriented(X,v.x,v.y,v.z));
  }
  if (oriented_only)
    return;
  // Verify that all internal edges are Delaunay, and all boundary vertices are convex
  for (const auto ee : mesh.edges()) {
    auto e = mesh.halfedge(ee,0);
    if (mesh.is_boundary(ee)) {
      e = mesh.is_boundary(e)?e:mesh.reverse(e);
      // Check convexity
      const auto v0 = mesh.src(e),
                 v1 = mesh.dst(e),
                 v2 = mesh.dst(mesh.next(e));
      OTHER_ASSERT(triangle_oriented(X,v2,v1,v0));
    } else if (!is_delaunay(mesh,X,e))
      throw RuntimeError(format("non delaunay edge: e%d, v%d v%d",e.id,mesh.src(e).id,mesh.dst(e).id));
  }
}

// This routine assumes the sentinel points have already been added
Ref<HalfedgeMesh> exact_delaunay_helper(RawField<const EV,VertexId> X, const bool validate) {
  const int n = X.size()-3;
  const auto mesh = new_<HalfedgeMesh>();
  IntervalScope scope;

  // Initialize the mesh to a Delaunay triangle containing the sentinels at infinity.
  mesh->add_vertices(n+3);
  mesh->add_face(vec(VertexId(n+0),VertexId(n+1),VertexId(n+2)));
  if (self_check)
    assert_delaunay(mesh,X);

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
    check_bsp(mesh,bsp,face_to_bsp,X);

  // Allocate a stack to simulate recursion when flipping non-Delaunay edges.
  // Invariant: if edge is on the stack, the other edges of face(edge) are Delaunay.
  Array<HalfedgeId> stack;
  stack.preallocate(8);

  // Insert all vertices into the mesh in random order, maintaining the Delaunay property
  for (const auto i : range(n)) {
    check_interrupts();
    // Pick a vertex at random, without replacement
    const VertexId v(random_permute(n,key,i));

    // Search through the BSP tree to find the containing triangle
    const auto f0 = bsp_search(bsp,X,v);

    // Split the face by inserting the new vertex and update the BSP tree accordingly.
    mesh->split_face(f0,v);
    const auto e0 = mesh->halfedge(v),
               e1 = mesh->left(e0),
               e2 = mesh->right(e0);
    const auto f1 = mesh->face(e1),
               f2 = mesh->face(e2);
    int base = bsp.size();
    bsp.resize(base+3,false);
    set_links(bsp,face_to_bsp[f0],base);
    bsp[base+0] = Node(vec(v,mesh->dst(e0)),base+2,base+1);
    bsp[base+1] = Node(vec(v,mesh->dst(e1)),~f0.id,~f1.id);
    bsp[base+2] = Node(vec(v,mesh->dst(e2)),~f1.id,~f2.id);
    face_to_bsp[f0] = vec(2*(base+1)+0,-1);
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+1)+1,2*(base+2)+0));
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+2)+1,-1));
    if (self_check)
      check_bsp(mesh,bsp,face_to_bsp,X);

    // Fix all non-Delaunay edges
    stack.resize(3,false,false); 
    stack[0] = mesh->next(e0);
    stack[1] = mesh->next(e1);
    stack[2] = mesh->next(e2);
    if (self_check)
      assert_delaunay(mesh,X,true);
    while (stack.size()) {
      const auto e = stack.pop();
      if (!is_delaunay(mesh,X,e)) {
        // Our mesh is linearly embedded in the plane, so edge flips are always safe
        assert(mesh->is_flip_safe(e));
        mesh->unsafe_flip_edge(e);
        OTHER_ASSERT(is_delaunay(mesh,X,e));
        // Update the BSP tree for the triangle flip
        const auto f0 = mesh->face(e),
                   f1 = mesh->face(mesh->reverse(e)); 
        const int node = bsp.append(Node(mesh->vertices(e),~f1.id,~f0.id));
        set_links(bsp,face_to_bsp[f0],node);
        set_links(bsp,face_to_bsp[f1],node);
        face_to_bsp[f0] = vec(2*node+1,-1);
        face_to_bsp[f1] = vec(2*node+0,-1);
        if (self_check)
          check_bsp(mesh,bsp,face_to_bsp,X);
        // Recurse to successor edges to e
        stack.append_elements(vec(mesh->next(e),mesh->prev(mesh->reverse(e))));
      }
      if (self_check)
        assert_delaunay(mesh,X,true);
    }
    if (self_check)
      assert_delaunay(mesh,X);
  }

  // Remove sentinels
  for (int i=0;i<3;i++)
    mesh->unsafe_delete_last_vertex();

  // If desired, check that the final mesh is Delaunay
  if (validate)
    assert_delaunay(mesh,X);

  // Return the mesh with the sentinels removed
  return mesh;
}

Ref<HalfedgeMesh> delaunay_points(RawArray<const Vector<real,2>> X_, bool validate) {
  const int n = X_.size();
  OTHER_ASSERT(n>=3);

  // Quantize all input points
  Field<EV,VertexId> X(n+3,false);
  X.flat.slice(0,n) = amap(quantizer(bounding_box(X_)),X_);

  // Add 3 sentinel points at infinity
  X.flat[n+0] = EV(-bound,-bound);
  X.flat[n+1] = EV(bound,0);
  X.flat[n+2] = EV(-bound,bound);

  // Compute Delaunay triangulation
  return exact_delaunay_helper(X,validate);
}

// Same as above, but points are already quantized
Ref<HalfedgeMesh> exact_delaunay_points(RawArray<const EV> X_, bool validate) {
  const int n = X_.size();
  OTHER_ASSERT(n>=3);

  // Quantize all input points
  Field<EV,VertexId> X(n+3,false);
  X.flat.slice(0,n) = X_;

  // Add 3 sentinel points at infinity
  X.flat[n+0] = EV(-bound,-bound);
  X.flat[n+1] = EV(bound,0);
  X.flat[n+2] = EV(-bound,bound);

  // Compute Delaunay triangulation
  return exact_delaunay_helper(X,validate);
}

}
using namespace other;

void wrap_delaunay() {
  OTHER_FUNCTION_2(delaunay_points_py,delaunay_points)
}
