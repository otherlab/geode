// Triangulation of segments in the plane using sweep and orientation tests.

#include <geode/exact/simple_triangulate.h>
#include <geode/exact/Interval.h>
#include <geode/exact/math.h>
#include <geode/exact/scope.h>
#include <geode/array/sort.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/utility/curry.h>
namespace geode {

// We construct a conservative interval based policy and use it to test
// triangulate_monotone_polygon and add_constraint_edge.
typedef Vector<double,2> TV;
typedef Vector<Interval,2> IV;
using std::cout;
using std::endl;

namespace {
struct Naive {
  Field<IV,VertexId> X;
  typedef Vector<VertexId,2> Line;

  Line reverse_line(const Line v) const {
    return v.reversed();
  }

  bool below(const VertexId v0, const VertexId v1) const {
    if (v0 == v1)
      return false;
    const int s = weak_sign(X[v1].y-X[v0].y);
    GEODE_ASSERT(s);
    return s>0;
  }

  bool triangle_oriented(const VertexId v0, const VertexId v1, const VertexId v2) const {
    GEODE_ASSERT(v0!=v1 && v1!=v2 && v2!=v0);
    const int s = weak_sign(edet(X[v1]-X[v0],X[v2]-X[v0]));
    GEODE_ASSERT(s);
    return s>0;
  }

  bool line_point_oriented(const Line v01, const VertexId v2) const {
    return triangle_oriented(v01.x,v01.y,v2);
  }

  VertexId construct_segment_intersection(MutableTriangleTopology& mesh, const Line a, const Line b) {
    // cross(a0 + t*(a1-a0) - b0, b1-b0) = 0
    // cross(a0 + t*da - b0, db) = 0
    // t*cross(da,db) = cross(b0-a0,db)
    const auto da = X[a.y]-X[a.x],
               db = X[b.y]-X[b.x];
    const auto den = cross(da,db);
    GEODE_ASSERT(!den.contains_zero());
    const auto t = cross(X[b.x]-X[a.x],db)*inverse(den);
    GEODE_ASSERT(certainly_positive(t) && certainly_less(t,1));
    // Add the new vertex to the mesh
    const auto u = X.append(X[a.x]+t*da),
               v = mesh.add_vertex();
    GEODE_ASSERT(u == v);
    return u;
  }
};
}

static void check(const Naive& P, const TriangleTopology& mesh, const int boundary) {
  // Verify that we have exactly one boundary contour with the expected size
  GEODE_ASSERT(mesh.n_boundary_edges()==boundary);
  GEODE_ASSERT(mesh.boundary_loops().size()==1);
  GEODE_ASSERT(mesh.is_manifold_with_boundary());
  GEODE_ASSERT(!mesh.has_isolated_vertices());

  // Verify that all vertices we expect to be boundary are
  for (const int v : range(boundary))
    GEODE_ASSERT(mesh.is_boundary(VertexId(v)));

  // Verify that all triangles are positively oriented
  for (const auto f : mesh.faces()) {
    const auto v = mesh.vertices(f);
    GEODE_ASSERT(P.triangle_oriented(v.x,v.y,v.z));
  }
}

static TV random_in_triangle(Random& random, const TV p0, const TV p1, const TV p2) {
  double t0 = random.uniform(),
         t1 = random.uniform();
  if (t0+t1 > 1) {
    t0 = 1-t0;
    t1 = 1-t1;
  }
  return t0*p0+t1*p1+(1-t0-t1)*p2;
}

static void simple_triangulate_test(const int seed, const int left, const int right,
                                                    const int interior, const int edges) {
  IntervalScope scope;
  const auto random = new_<Random>(seed);
  const int boundary = 2+left+right;

  // Pick a bunch of boundary points on the unit circle
  Naive P;
  const auto lo = P.X.append(IV(0,-1));
  const auto hi = P.X.append(IV(0,+1));
  Array<VertexId> lefts, rights;
  for (const int i : range(left))
    lefts.append (P.X.append(IV(polar(-pi/2-pi/(left +1)*(i+1)))));
  for (const int i : range(right))
    rights.append(P.X.append(IV(polar(-pi/2+pi/(right+1)*(i+1)))));

  // Pick a bunch of points in the interior of the boundary polygon, then sort upwards.
  Array<VertexId> interiors;
  const double left_area  = 2*(1+left )*sin(pi/(1+left)),
               right_area = 2*(1+right)*sin(pi/(1+right)),
               left_fraction = left_area/(left_area+right_area);
  for (int i=0;i<interior;i++) {
    if (random->uniform() < left_fraction) {
      const int j = random->uniform<int>(0,1+left);
      interiors.append(P.X.append(IV(random_in_triangle(random,TV(),center(P.X[j?lefts[j-1]:lo]),
                                                                    center(P.X[j<left?lefts[j]:hi])))));
    } else {
      const int j = random->uniform<int>(0,1+right);
      interiors.append(P.X.append(IV(random_in_triangle(random,TV(),center(P.X[j?rights[j-1]:lo]),
                                                                    center(P.X[j<right?rights[j]:hi])))));
    }
  }
  sort(interiors,curry(&Naive::below,&P));

  // Print points
  if (0) {
    Array<TV> Xc;
    for (const auto x : P.X.flat)
      Xc.append(center(x));
    cout << "X = "<<Xc<<endl;
  }

  // Triangulate all points as two monotone polygons
  const auto mesh = new_<MutableTriangleTopology>();
  mesh->add_vertices(P.X.size());
  Array<VertexId> stack;
  triangulate_monotone_polygon(P,mesh,lo,hi,lefts, interiors,stack);
  triangulate_monotone_polygon(P,mesh,lo,hi,interiors,rights,stack);
  check(P,mesh,boundary);

  // Print triangles
  if (0)
    cout << "\ntris = "<<mesh->elements()<<endl;

  // Pick a bunch of constraint edges, ignoring self intersections but avoiding duplicates
  const int n = P.X.size();
  Hashtable<Vector<VertexId,2>> used;
  Hashtable<Vector<VertexId,2>,Vector<VertexId,2>> constrained;
  int added = 0;
  for (int i=0;i<edges;i++) {
    const VertexId v0(random->uniform<int>(0,n)),
                   v1(random->uniform<int>(0,n));
    if (v0!=v1 && used.set(vec(v0,v1).sorted())) {
      add_constraint_edge(P,mesh,constrained,v0,v1,vec(v0,v1));
      added++;
    }
  }
  check(P,mesh,boundary);

  // Verify that we have the right number of constrained edges
  int count = 0;
  for (const auto e : mesh->halfedges()) {
    const auto v = mesh->vertices(e);
    if (v.x<v.y && constrained.contains(v))
      count++;
  }
  GEODE_ASSERT(count==added+2*(P.X.size()-n));
  GEODE_ASSERT(constrained.size()==added+3*(P.X.size()-n));
}

}
using namespace geode;

void wrap_simple_triangulate() {
  GEODE_FUNCTION(simple_triangulate_test)
}
