// Randomized incremental Delaunay using simulation of simplicity

#include <other/core/exact/delaunay.h>
#include <other/core/exact/predicates.h>
#include <other/core/exact/quantize.h>
#include <other/core/exact/scope.h>
#include <other/core/array/amap.h>
#include <other/core/array/RawField.h>
#include <other/core/math/integer_log.h>
#include <other/core/python/wrap.h>
#include <other/core/random/permute.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
namespace other {

using Log::cout;
using std::endl;
typedef Vector<real,2> TV;
typedef Vector<Quantized,2> EV;
typedef exact::Point2 Point;
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
//
// Randomized incremental Delaunay by itself has terrible cache properties, since the mesh is accessed in nearly purely
// random order.  Therefore, we use the elegant partial randomization idea of
//
//    Nina Amenta, Sunghee Choi, Gunter Rote, "Incremental Constructions con BRIO".
//
// which randomly assigns vertices into bins of exponentially increasing size, then applies a spatial sort within each bin.
// The random choice of bin is sufficient to ensure an O(n log n) worst case time, and the spatial sort produces a relatively
// cache coherent access pattern.

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

static inline bool is_sentinel(const Point& x) {
  return abs(x.y.x)==bound;
}

// Different sentinels will be placed at different orders of infinity, with earlier indices further away.
// These macros are used in the sorting networks below to move large sentinels last
#define SENTINEL_KEY(i) (is_sentinel(x##i)?-x##i.x:numeric_limits<int>::min())
#define COMPARATOR(i,j) if (SENTINEL_KEY(i)>SENTINEL_KEY(j)) { swap(x##i,x##j); parity ^= 1; }

OTHER_COLD OTHER_CONST static inline bool triangle_oriented_sentinels(Point x0, Point x1, Point x2) {
  // Move large sentinels last
  bool parity = 0;
  COMPARATOR(0,1)
  COMPARATOR(1,2)
  COMPARATOR(0,1)
  // Triangle orientation tests reduce to segment vs. direction and direction vs. direction when infinities are involved
  return parity ^ (!is_sentinel(x1) ? segment_to_direction_oriented(x0,x1,x2)   // one sentinel
                                    :           directions_oriented(   x1,x2)); // two or three sentinels
}

OTHER_ALWAYS_INLINE static inline bool triangle_oriented(const RawField<const Point,VertexId> X, VertexId v0, VertexId v1, VertexId v2) {
  const auto x0 = X[v0],
             x1 = X[v1],
             x2 = X[v2];
  // If we have a nonsentinel triangle, use a normal orientation test
  if (maxabs(x0.y.x,x1.y.x,x2.y.x)!=bound)
    return triangle_oriented(x0,x1,x2);
  // Fall back to sentinel case analysis
  return triangle_oriented_sentinels(x0,x1,x2);
}

// Test whether an edge containing sentinels is Delaunay
OTHER_COLD OTHER_CONST static inline bool is_delaunay_sentinels(Point x0, Point x1, Point x2, Point x3) {
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
template<class Mesh> OTHER_ALWAYS_INLINE static inline bool is_delaunay(const Mesh& mesh, RawField<const Point,VertexId> X, const HalfedgeId edge) {
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
  if (maxabs(x0.y.x,x1.y.x,x2.y.x,x3.y.x)!=bound)
    return !incircle(x0,x1,x2,x3);
  // Fall back to sentinel case analysis
  return is_delaunay_sentinels(x0,x1,x2,x3);
}

static inline FaceId bsp_search(RawArray<const Node> bsp, RawField<const Point,VertexId> X, const VertexId v) {
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

template<class Mesh> OTHER_UNUSED static void check_bsp(const Mesh& mesh, RawArray<const Node> bsp, RawField<const Vector<int,2>,FaceId> face_to_bsp, RawField<const Point,VertexId> X_) {
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
      const auto center = X.append(tuple(X.size(),(X[v.x].y+X[v.y].y+X[v.z].y)/3));
      const auto found = bsp_search(bsp,X,center);
      if (found!=f) {
        cout << "bsp search failed: f "<<f<<", v "<<v<<", found "<<found<<endl;
        OTHER_ASSERT(false);
      }
      X.flat.pop();
    }
  }
}

OTHER_COLD static void assert_delaunay(const CornerMesh& mesh, RawField<const Point,VertexId> X, const bool oriented_only=false) {
  // Verify that all faces are correctly oriented
  for (const auto f : mesh.faces()) {
    const auto v = mesh.vertices(f);
    OTHER_ASSERT(triangle_oriented(X,v.x,v.y,v.z));
  }
  if (oriented_only)
    return;
  // Verify that all internal edges are Delaunay
  for (const auto e : mesh.interior_halfedges())
    if (!mesh.is_boundary(mesh.reverse(e)) && mesh.src(e)<mesh.dst(e))
      if (!is_delaunay(mesh,X,e))
        throw RuntimeError(format("non delaunay edge: e%d, v%d v%d",e.id,mesh.src(e).id,mesh.dst(e).id));
  // Verify that all boundary vertices are convex
  for (const auto e : mesh.boundary_edges()) {
    const auto v0 = mesh.src(mesh.prev(e)),
               v1 = mesh.src(e),
               v2 = mesh.dst(e);
    OTHER_ASSERT(triangle_oriented(X,v2,v1,v0));
  }
}

OTHER_UNUSED OTHER_COLD static void assert_delaunay(const HalfedgeMesh& mesh, RawField<const Point,VertexId> X, const bool oriented_only=false) {
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

static inline void unsafe_flip_edge(CornerMesh& mesh, HalfedgeId& e) {
  e = mesh.unsafe_flip_edge(e);
}

static inline void unsafe_flip_edge(HalfedgeMesh& mesh, const HalfedgeId e) {
  mesh.unsafe_flip_edge(e);
}

// This routine assumes the sentinel points have already been added, and processes points in order
template<class Mesh> OTHER_NEVER_INLINE static Ref<Mesh> deterministic_exact_delaunay(RawField<const Point,VertexId> X, const bool validate) {
  const int n = X.size()-3;
  const auto mesh = new_<Mesh>();
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
    int base = bsp.size();
    bsp.resize(base+3,false);
    set_links(bsp,face_to_bsp[f0],base);
    bsp[base+0] = Node(vec(v,vs.x),base+2,base+1);
    bsp[base+1] = Node(vec(v,vs.y),~f0.id,~f1.id);
    bsp[base+2] = Node(vec(v,vs.z),~f1.id,~f2.id);
    face_to_bsp[f0] = vec(2*(base+1)+0,-1);
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+1)+1,2*(base+2)+0));
    face_to_bsp.flat.append_assuming_enough_space(vec(2*(base+2)+1,-1));
    if (self_check)
      check_bsp(*mesh,bsp,face_to_bsp,X);

    // Fix all non-Delaunay edges
    stack.resize(3,false,false);
    stack[0] = tuple(mesh->next(e0),vec(vs.x,vs.y));
    stack[1] = tuple(mesh->next(e1),vec(vs.y,vs.z));
    stack[2] = tuple(mesh->next(e2),vec(vs.z,vs.x));
    if (self_check)
      assert_delaunay(mesh,X,true);
    while (stack.size()) {
      const auto evs = stack.pop();
      auto e = mesh->vertices(evs.x)==evs.y ? evs.x : mesh->halfedge(evs.y.x,evs.y.y);
      if (e.valid() && !is_delaunay(*mesh,X,e)) {
        // Our mesh is linearly embedded in the plane, so edge flips are always safe
        assert(mesh->is_flip_safe(e));
        unsafe_flip_edge(mesh,e);
        OTHER_ASSERT(is_delaunay(*mesh,X,e));
        // Update the BSP tree for the triangle flip
        const auto f0 = mesh->face(e),
                   f1 = mesh->face(mesh->reverse(e));
        const int node = bsp.append(Node(mesh->vertices(e),~f1.id,~f0.id));
        set_links(bsp,face_to_bsp[f0],node);
        set_links(bsp,face_to_bsp[f1],node);
        face_to_bsp[f0] = vec(2*node+1,-1);
        face_to_bsp[f1] = vec(2*node+0,-1);
        if (self_check)
          check_bsp(*mesh,bsp,face_to_bsp,X);
        // Recurse to successor edges to e
        const auto e0 = mesh->next(e),
                   e1 = mesh->prev(mesh->reverse(e));
        stack.extend(vec(tuple(e0,mesh->vertices(e0)),
                         tuple(e1,mesh->vertices(e1))));
        if (self_check)
          assert_delaunay(mesh,X,true);
      }
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

template<int axis> static inline int spatial_partition(RawArray<Point> X, Random& random) {
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

static void spatial_sort(RawArray<Point> X, const int leaf_size, Random& random) {
  const int n = X.size();
  if (n<=leaf_size)
    return;

  // We determine the subdivision axis using inexact computation, which is okay since neither the result nor
  // the asymptotic worst case complexity depends upon any properties of the spatial_sort whatsoever.
  const Box<EV> box = bounding_box(X.project<EV,&Point::y>());
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
template<class Inputs> static Array<Point> partially_sorted_shuffle(const Inputs& Xin) {
  const int n = Xin.size();
  Array<Point> X(n+3,false);

  // Randomly assign input points into bins.  Bin k has 2**k = 1,2,4,8,... and starts at index 2**k-1 = 0,1,3,7,...
  // We fill points into bins as sequentially as possible to maximize cache coherence.
  const int bins = integer_log(n);
  Array<int> bin_counts(bins);
  #pragma omp parallel for
  for (int i=0;i<n;i++) {
    int j = random_permute(n,key,i);
    const int bin = min(integer_log(j+1),bins-1);
    j = (1<<bin)-1+bin_counts[bin]++;
    X[j] = tuple(i,Xin[i]);
  }

  // Spatially sort each bin down to clusters of size 64.
  const int leaf_size = 64;
  #pragma omp parallel for
  for (int bin=0;bin<bins;bin++) {
    const int start = (1<<bin)-1,
              end = bin==bins-1?n:start+(1<<bin);
    assert(bin_counts[bin]==end-start);
    spatial_sort(X.slice(start,end),leaf_size,new_<Random>(key+bin));
  }

  // Add 3 sentinel points at infinity
  X[n+0] = tuple(n+0,EV(-bound,-bound));
  X[n+1] = tuple(n+1,EV( bound, 0)    );
  X[n+2] = tuple(n+2,EV(-bound, bound));

  return X;
}

template<class Mesh,class Inputs> static inline Ref<Mesh> delaunay_helper(const Inputs& Xin, bool validate) {
  const int n = Xin.size();
  OTHER_ASSERT(n>=3);

  // Quantize all input points, reorder, and add sentinels
  Field<const Point,VertexId> X(partially_sorted_shuffle(Xin));

  // Compute Delaunay triangulation
  const auto mesh = deterministic_exact_delaunay<Mesh>(X,validate);

  // Undo the vertex permutation
  mesh->permute_vertices(X.flat.slice(0,n).project<int,&Point::x>().copy());
  return mesh;
}

template<class Mesh> Ref<Mesh> delaunay_points(RawArray<const Vector<real,2>> X, bool validate) {
  return delaunay_helper<Mesh>(amap(quantizer(bounding_box(X)),X),validate);
}

// Same as above, but points are already quantized
template<class Mesh> Ref<Mesh> exact_delaunay_points(RawArray<const EV> X, bool validate) {
  return delaunay_helper<Mesh>(X,validate);
}

}
using namespace other;

void wrap_delaunay() {
  OTHER_FUNCTION_2(delaunay_points_corner,delaunay_points<CornerMesh>)
  OTHER_FUNCTION_2(delaunay_points_halfedge,delaunay_points<HalfedgeMesh>)
}
