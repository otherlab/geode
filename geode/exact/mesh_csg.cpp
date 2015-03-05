// Robust constructive solid geometry for triangle meshes

#include <geode/exact/mesh_csg.h>
#include <geode/exact/debug.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/exact/predicates.h>
#include <geode/exact/quantize.h>
#include <geode/exact/scope.h>
#include <geode/exact/simple_triangulate.h>
#include <geode/array/amap.h>
#include <geode/array/ConstantMap.h>
#include <geode/array/RawField.h>
#include <geode/array/reversed.h>
#include <geode/array/sort.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/geometry/traverse.h>
#include <geode/math/mean.h>
#include <geode/math/optimal_sort.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/python/function.h>
#include <geode/python/wrap.h>
#include <geode/random/permute.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/structure/UnionFind.h>
#include <geode/utility/Unique.h>
#include <geode/vector/Matrix.h>
namespace geode {

// Algorithm explanation:
//
// Mesh CSG proceeds in two steps: face splitting and depth computation.  In the first step,
// we break each face of the original mesh into subfaces along other faces of the mesh.  This
// produces an intersection free nonmanifold mesh.  In the second step, we compute the depth
// of each subface using a combination of ray casting and flood fill.  If we are doing normal
// CSG, we can then discard all subfaces of the wrong depth and stitch those of correct depth
// into a closed mesh.
//
// In order to split faces, we must compute all intersection edges and vertices.  There are
// two kinds of intersection vertices: new vertices defined by the intersection of a disjoint
// edge-face pair, and existing "loop" vertices which are the endpoint of an intersection edge
// between two faces that share exactly one vertex.  Both ends of each intersection edge have
// this form.  Moreover, all loop vertices are at the other end of an intersection edge with
// a normal edge-face intersection vertex.  Therefore, we can compute all the intersection
// information we need using a single edge-face box hierarchy traversal.

typedef exact::Vec3 EV;
typedef Vector<double,3> TV;
typedef Vector<Interval,3> IV;
typedef exact::Perturbed3 P;
using std::cout;
using std::endl;

// For interface simplicity, we use a single fixed random number as the seed.
// This is safe unless the points are chosen maliciously.  We've reused the key
// from delaunay.cpp.
static const uint128_t key = 9975794406056834021u+(uint128_t(920519151720167868u)<<64);

// Some variables in the code can refer to either original vertices (possibly loop vertices),
// edge-face intersection vertices, or face-face-face intersection vertices.  For this purpose,
// we concatenate the ranges for the three types of vertices in order.  The same numbering is
// used for vertices in the split output mesh before pruning.
#define Xi(i) P(i,X[i])
#define Xi2(i) (i < X.size() ? IV(X[i]) \
                             : ef_vertices.flat[i-X.size()].p())
#define Xi3(i) (  i < X.size()                    ? IV(X[i]) \
                : i-X.size() < ef_vertices.flat.size() ? ef_vertices.flat[i-X.size()].p() \
                                                       : fff_vertices[i-X.size()-ef_vertices.flat.size()].p())

// Convenience macros for passing and declaring many arguments at once
#define VA(v) const TV v
#define EA(e) const TV e##0, const TV e##1
#define FA(f) const TV f##0, const TV f##1, const TV f##2
#define EX(e) Xi(e.x),Xi(e.y)
#define FX(f) Xi(f.x),Xi(f.y),Xi(f.z)
#define FX0(f) f.x,f.y,f.z

static inline IV iv(const P& p) { return IV(p.value()); }
// Constructed points are guaranteed to be within this tolerance
const Quantized tolerance = 1;

namespace {
struct Vertex {
  // Guaranteed within tolerance of the exact value
  EV rounded;

  // A conservative interval containing the true intersection
  IV p() const {
    return IV(Interval(rounded.x-tolerance,rounded.x+tolerance),
              Interval(rounded.y-tolerance,rounded.y+tolerance),
              Interval(rounded.z-tolerance,rounded.z+tolerance));
  }
};

struct EdgeFaceVertex : public Vertex {
  int edge;
  int face;
  bool flip; // False if the edge and face have compatible orientations

  EdgeFaceVertex() = default;

  EdgeFaceVertex(const int edge, const int face, const bool flip, const EV rounded)
    : Vertex({rounded}), edge(edge), face(face), flip(flip) {}
};

struct FaceFaceFaceVertex : public Vertex {
  Vector<int,3> faces; // f0,f1,f2 ordered so that det(n0,n1,n2) > 0

  FaceFaceFaceVertex(const Vector<int,3> faces, const EV rounded)
    : Vertex({rounded}), faces(faces) {}
};

struct FaceFaceEdge {
  Vector<int,2> faces; // Invariant: cross(normal(faces.x),normal(faces.y)) . (nodes.y-nodes.x) > 0
  Vector<int,2> nodes; // May be loop or edge-face vertices
};

// Construct an edge-face vertex
struct ConstructEF {
  // Our constructed point p is given by
  //   p = e0+t(e1-e0)
  //   dot(n,e0+t(e1-e0)-f0) = 0
  //   t dot(n,e1-e0) + dot(n,e0-f0) = 0
  //   t = dot(n,f0-e0) / dot(n,e1-e0)
  template<class TV> static inline ConstructType<3,4,TV> eval(EA(e),FA(f)) {
    const auto n = ecross(f1-f0,f2-f0);
    const auto de = e1-e0;
    const auto a = edot(n,f0-e0),
               b = edot(n,de);
    return tuple(emul(b,e0)+emul(a,de),b);
  }
};

// Construct a face-face-face vertex
struct ConstructFFF {
  template<class TV> static inline ConstructType<3,7,TV> eval(FA(f0),FA(f1),FA(f2)) {
    // Construct a 3x3 linear system out of the face normals and solve
    #define N(i) const auto n##i = ecross(f##i##1-f##i##0,f##i##2-f##i##0);
    #define R(i) const auto r##i = edot(n##i,f##i##0);
    #define C(i,i1,i2,j,j1,j2) const auto c##i##j = n##j1[i1]*n##j2[i2]-n##j1[i2]*n##j2[i1];
    N(0) N(1) N(2)
    R(0) R(1) R(2)
    C(0,1,2, 0,1,2) C(0,1,2, 1,2,0) C(0,1,2, 2,0,1)
    C(1,2,0, 0,1,2) C(1,2,0, 1,2,0) C(1,2,0, 2,0,1)
    C(2,0,1, 0,1,2) C(2,0,1, 1,2,0) C(2,0,1, 2,0,1)
    #define P(i) (c##i##0*r##0+c##i##1*r##1+c##i##2*r##2)
    const auto p = vec(P(0),P(1),P(2));
    const auto q = n0.x*c00+n0.y*c10+n0.z*c20;
    return tuple(p,q); // p/q
    #undef N
    #undef R
    #undef C
    #undef P
  }
};
}

// True if triangle f contains edge e with negative orientation, false for positive.  Asserts that e in f.
static inline bool flipped_in(const Vector<int,2> e, const Vector<int,3> f) {
  assert(f.contains_all(e));
  return e.y != (  e.x==f.x ? f.y
                 : e.x==f.y ? f.z
                            : f.x);
}

// Find the other two entries after a given one in the cyclic order
static inline Vector<int,2> next_two(const Vector<int,3> v, const int i) {
  assert(v.contains(i));
  return i==v.x ? vec(v.y,v.z)
       : i==v.y ? vec(v.z,v.x)
                : vec(v.x,v.y);
}

namespace {
// State management during face retriangulation.
// This class is primarily responsible for maintaing face-face-face vertices
struct State {
  // Vertices of all three types
  const RawArray<const EV> X;
  const Nested<const EdgeFaceVertex> ef_vertices;
  Array<FaceFaceFaceVertex>& fff_vertices;
  Hashtable<Vector<int,3>,int>& faces_to_fff; // Sorted faces to corresponding fff vertex

  // Topology
  const RawArray<const Vector<int,3>> faces;
  const RawArray<const Vector<int,2>> edges;

  // depth weights for faces
  const RawArray<const int> depth_weight;

  State(RawArray<const EV> X, Nested<const EdgeFaceVertex> ef_vertices,
        Array<FaceFaceFaceVertex>& fff_vertices, Hashtable<Vector<int,3>,int>& faces_to_fff,
        RawArray<const Vector<int,3>> faces, RawArray<const Vector<int,2>> edges, RawArray<const int> depth_weight)
    : X(X)
    , ef_vertices(ef_vertices)
    , fff_vertices(fff_vertices)
    , faces_to_fff(faces_to_fff)
    , faces(faces)
    , edges(edges)
    , depth_weight(depth_weight)
    {}

  // Is vertex v above face f?
  bool face_vertex_oriented(const int f, const int v) const {
    const auto fv = faces[f];
    const P f0 = Xi(fv.x),
            f1 = Xi(fv.y),
            f2 = Xi(fv.z);
    const auto q = Xi3(v),
               i0 = iv(f0),
               i1 = iv(f1),
               i2 = iv(f2);
    return FILTER(edet(i1-i0,i2-i0,q-i0),
                  face_vertex_oriented_helper(f0,f1,f2,v));
  }

  // Given face (f0,f1,f2), edge (e0,e1) and face (t0,t1,t2), let q = intersection((e0,e1),(t0,t1,t2)).
  // Is (f0,f1,f2,q) oriented?
  struct OrientFEF {
    template<class TV> static PredicateType<6,TV> eval(FA(f),EA(e),FA(t)) {
      const auto de = e1-e0;
      const auto nf = ecross(f1-f0,f2-f0),
                 nt = ecross(t1-t0,t2-t0);
      return edot(nt,de)*edot(nf,e0-f0)-edot(nf,de)*edot(nt,e0-t0);
    }
  };

  // Given triangles f0,f1,f2,f3, is f123 above f0?  If f123 is negatively oriented, the result is flipped.
  // This predicate is antisymmetric.
  struct OrientFFFF {
    template<class TV> static inline PredicateType<9,TV> eval(FA(f0),FA(f1),FA(f2),FA(f3)) {
      const auto n = ecross(f01-f00,f02-f00);
      const auto p = ConstructFFF::eval(f10,f11,f12,f20,f21,f22,f30,f31,f32);
      return edot(n,p.x-emul(p.y,f00));
    }
  };

  bool face_vertex_oriented_helper(const P f0, const P f1, const P f2, const int v) const {
    const int n = X.size(),
              nn = n+ef_vertices.flat.size();
    if (v < n) // Loop vertex
      return tetrahedron_oriented(f0,f1,f2,Xi(v));
    else if (v < nn) { // Edge-face vertex
      const auto& i = ef_vertices.flat[v-n];
      const P e0 = Xi(edges[i.edge].x),
              e1 = Xi(edges[i.edge].y),
              t0 = Xi(faces[i.face].x),
              t1 = Xi(faces[i.face].y),
              t2 = Xi(faces[i.face].z);
      return i.flip ^ perturbed_predicate<OrientFEF>(f0,f1,f2,e0,e1,t0,t1,t2);
    } else { // Face-face-face vertex
      const auto& i = fff_vertices[v-nn];
      const auto n0 = faces[i.faces.x],
                 n1 = faces[i.faces.y],
                 n2 = faces[i.faces.z];
      return perturbed_predicate<OrientFFFF>(f0,f1,f2,FX(n0),FX(n1),FX(n2));
    }
  }
};

// Information about a line during retriangulation
struct Line {
  int face;
  int ff; // ff-edge index if we're positively oriented, -ff-1 if we're negatively oriented
};

// Geometry policy for triangulation within the given face
struct Policy : public State, public Noncopyable {
  const int face;
  const Vector<P,3> f;
  const Vector<int,3> face_edges; // e12,e20,e01
  const IV normal;
  Hashtable<Vector<VertexId,2>,Line> constrained;

  // Contiguous list of vertices for this face
  Field<int,VertexId> vertices;

  Policy(State& S, const int face, const Vector<int,3> face_edges)
    : State(S)
    , face(face)
    , f(Xi(faces[face].x),
        Xi(faces[face].y),
        Xi(faces[face].z))
    , face_edges(face_edges)
    , normal(cross(iv(f.y)-iv(f.x),iv(f.z)-iv(f.x))) {
    assert(edges[face_edges.z].contains_all(vec(f.x.seed(),f.y.seed())));
  }

  Line reverse_line(const Line L) const {
    return Line({L.face,-L.ff-1});
  }

  bool line_point_oriented(const Line L, const VertexId v) const {
    return (L.ff<0) ^ face_vertex_oriented(L.face,vertices[v]);
  }

  bool triangle_oriented(const VertexId v0, const VertexId v1, const VertexId v2) const {
    const int i0 = vertices[v0],
              i1 = vertices[v1],
              i2 = vertices[v2];
    const auto x0 = Xi3(i0),
               x1 = Xi3(i1),
               x2 = Xi3(i2);
    return FILTER(dot(normal,cross(x1-x0,x2-x0)),
                  triangle_oriented_helper(i0,i1,i2));
  }

  // Let p2 be the point on e2 lying on the triangle.  Let n be the triangle normal.  We have
  //   p2 = e20 + t2*de2
  //   n.(p2-f0) = 0
  //   n.(e20+t2*de2-f0) = 0
  //   t2*n.de2 = n.(f0-e20)
  //   t2 = n.(f0-e20)/n.de2            - degree 3/3
  // Our predicate is
  //     flip1 * orient(v0,e11,e10,p2)
  //   = flip1 * det(e11-v0,e10-v0,p2-v0)
  //   = flip1 * det(e11-v0,e10-v0,e20-v0 + t2*de2)
  //   = flip1 * flip2 * det(e11-v0,e10-v0,(n.de2)*(e20-v0)+(n.(f0-e20))*de2)
  //   = flip1 * flip2 * det(e11-v0,e10-v0,(n.de2)*(e20-v0)-(n.(e20-f0))*de2)
  #define EF_TERM(base,i) (  emul(edot(n##i,d##i),e##i##0-(base)) \
                           - emul(edot(n##i,e##i##0-f##i##0),d##i))
  struct OrientVEFEF { template<class TV> static PredicateType<6,TV> eval(VA(v0),EA(e1),EA(e2),FA(f2)) {
      const auto n2 = ecross(f21-f20,f22-f20);
      const auto d2 = e21-e20;
      return edot(ecross(e11-v0,e10-v0),EF_TERM(v0,2));
  }};

  // Define n,ti,pi as above.  Let xi = ei0-e00, yi = ei0-v0.  We have
  //   pi = ei0 + ti*dei = ei0 - (n.yi)dei/n.dei
  // Our predicate is
  //     flip0 * orient(e00,e01,p1,p2)
  //   = flip0 * det(x0,x1+t1*de1,x2+t2*de2)
  //   = flip0 * flip1 * flip2 * det(x0,(n.de1)*x1-(n.y1)*de1,
  //                                    (n.de2)*x2-(n.y2)*de2)
  // The following implementation allows the faces for edges 1 and 2 to be different.
  struct OrientEFEFEF { template<class TV> static PredicateType<9,TV> eval(EA(e0),EA(e1),FA(f1),EA(e2),FA(f2)) {
    const auto n1 = ecross(f11-f10,f12-f10),
               n2 = ecross(f21-f20,f22-f20);
    const auto d0 = e01-e00,
               d1 = e11-e10,
               d2 = e21-e20;
    return edot(ecross(d0,EF_TERM(e00,1)),EF_TERM(e00,2));
  }};

  // Assuming all positive suborientations, is (f0-f1,e1),(f0-f2,e2),(f0,f1,f2) oriented?
  #define N(f) const auto n##f = ecross(f##1-f##0,f##2-f##0);
  // dot(nei,f00-ei0+ti*(f0i-f00)) = 0
  // dot(nei,f00-ei0)+ti*dot(nei,f0i-f00) = 0
  // ti = dot(nei,ei0-f00) / dot(nei,f0i-f00) = ai/bi
  #define E(i) const auto a##i = edot(ne##i,e##i##0-f00), \
                          b##i = edot(ne##i,f0##i-f00);
  // Linear system for fff barycentric coordinates
  #define R(i) const auto r##i = edot(nf##i,f##i##0-f00);
  #define C(i,j) const auto c##i##j = edot(nf##i,f0##j-f00);
  // s = (u1,u2)/v
  #define U(i,ii) const auto u##i = c##ii##ii*r##i-c##i##ii*r##ii;
  #define V() const auto v = c11*c22-c12*c21;
  struct OrientBBFFF { template<class TV> static PredicateType<12,TV> eval(FA(e1),FA(e2),FA(f0),FA(f1),FA(f2)) {
    N(e1) N(e2) N(f1) N(f2)
    E(1) E(2)
    R(1) R(2) C(1,1) C(1,2) C(2,1) C(2,2)
    U(1,2) U(2,1) V()
    // P = orient((t1,0),(0,t2),s)
    //   = det((-t1,t2),s-(t1,0))
    //   = det((-t1,t2),(s1-t1,s2))
    //   = -t1*s2 - t2*(s1-t1)
    //   = t1*t2 - t1*s2 - t2*s1
    //   = a1*a2/(b1*b2) - a1*u2/(b1*v) - a2*u1/(b2*v)
    //   = (a1*a2*v - a1*b2*u2 - a2*b1*u1) / (b1*b2*v)
    return a1*(a2*v-b2*u2)-a2*b1*u1;
  }};
  // Same as OrientBBFFF, but the e2 intersection is exactly at f02.
  struct OrientBVFFF { template<class TV> static PredicateType<9,TV> eval(FA(e1),FA(f0),FA(f1),FA(f2)) {
    N(e1) N(f1) N(f2)
    E(1) // t2 = 1
    R(1) R(2) C(1,1) C(1,2) C(2,1) C(2,2)
    U(1,2) U(2,1) V()
    return a1*(v-u2)-b1*u1;
  }};
  #undef N
  #undef E
  #undef C
  #undef R
  #undef U

  // Assuming all positive suborientations, is (e0,_),(e1,f1),(f2,f3,f4) oriented?
  struct OrientEFEFFFF { template<class TV> static PredicateType<12,TV>
    eval(EA(e0),EA(e1),FA(f1),FA(f2),FA(f3),FA(f4)) {
      const auto d0 = e01-e00,
                 d1 = e11-e10;
      const auto n1 = ecross(f11-f10,f12-f10);
      const auto f = ConstructFFF::eval(f20,f21,f22,f30,f31,f32,f40,f41,f42);
      return edot(ecross(d0,EF_TERM(e00,1)),f.x-emul(f.y,e00));
  }};

  // Assuming all positive suborientations, is (e0,_),(f0,f1,f2),(f0,f3,f4) oriented?
  struct OrientEFFFFFFF { template<class TV> static PredicateType<15,TV>
    eval(EA(e0),FA(f0),FA(f1),FA(f2),FA(f3),FA(f4)) {
      const auto f = ConstructFFF::eval(f00,f01,f02,f10,f11,f12,f20,f21,f22),
                 g = ConstructFFF::eval(f00,f01,f02,f30,f31,f32,f40,f41,f42);
      return edot(ecross(e01-e00,f.x-emul(f.y,e00)),
                                 g.x-emul(g.y,e00));
  }};

  bool triangle_oriented_helper(int v0, int v1, int v2) const {
    // This routine is a huge case analysis on the different kinds of vertex patterns.
    // The types are V (input), B (boundary edge-face), EF (interior edge-face), and FFF (face-face-face).
    static const int V = 0, B = 1, EF = 2, FFF = 3;
    const auto fv = vec(f.x.seed(),f.y.seed(),f.z.seed());

    // Classify vertices
    const int n = X.size(),
              nn = n+ef_vertices.flat.size();
    #define TYPE(i) \
      (  v##i<n  ? V \
       : v##i<nn ? ef_vertices.flat[v##i-n].face==face ? EF : B \
                 : FFF)
    int t0 = TYPE(0),
        t1 = TYPE(1),
        t2 = TYPE(2);
    #undef TYPE

    // Sort simpler vertices first
    bool flip = false;
    #define C(i,j) \
      if (t##i > t##j) { \
        swap(t##i,t##j); \
        swap(v##i,v##j); \
        flip ^= 1; \
      }
    #define NOP()
    GEODE_SORT_NETWORK(3,C,NOP)
    #undef C

    #define C(t0,t1,t2) (9*(t0)+3*(t1)+(t2))
    switch (C(t0,t1,t2)) {
      case C(V,V,V):
        return flip ^ flipped_in(vec(v0,v1),fv) ^ 1;
      case C(V,V,B):
      case C(V,V,EF):
      case C(V,V,FFF): {
        // If we're not colinear, we're unconditionally positively oriented as long
        // as v0,v1 is oriented within the triangle.  If we're colinear, pretend we're
        // positively oriented as well to avoid sliver triangles.
        return flip ^ flipped_in(vec(v0,v1),fv) ^ 1; }
      case C(V,B,B): {
        const auto &ef1 = ef_vertices.flat[v1-n],
                   &ef2 = ef_vertices.flat[v2-n];
        auto e1 = edges[ef1.edge];
        if (ef1.edge == ef2.edge) {
          // v1,v2 are on the same edge.  Since edges are slightly concave in to prevent
          // sliver triangles, we're positively oriented iff v0 is the opposite vertex.
          return flip ^ flipped_in(e1,fv) ^ (v1<v2) ^ e1.contains(v0);
        } else {
          auto e2 = edges[ef2.edge];
          if (flipped_in(e1,fv)) swap(e1.x,e1.y);
          if (flipped_in(e2,fv)) swap(e2.x,e2.y);
          return flip ^ (e1.contains(v0) && e2.contains(v0)) ^ (e1.y==e2.x);
        }}
      case C(B,B,B): {
        const auto &ef0 = ef_vertices.flat[v0-n],
                   &ef1 = ef_vertices.flat[v1-n],
                   &ef2 = ef_vertices.flat[v2-n];
        const auto e0 = edges[ef0.edge],
                   e1 = edges[ef1.edge];
        return flip ^ (  ef0.edge==ef1.edge ?   flipped_in(e0,fv) ^ (v0<v1)
                                              ^ (ef0.edge==ef2.edge ? (v0>v2) ^ (v1<v2) : 0)
                       : ef0.edge==ef2.edge ? flipped_in(e0,fv) ^ (v2<v0)
                       : ef1.edge==ef2.edge ? flipped_in(e1,fv) ^ (v1<v2)
                                            : flipped_in(vec(ef1.edge,ef0.edge),face_edges)); }
      case C(V,B,EF):
      case C(V,B,FFF): {
        const auto& ef1 = ef_vertices.flat[v1-n];
        const auto e1 = edges[ef1.edge];
        if (e1.contains(v0)) {
          // v1 is on an edge which touches v0, so we're the same as above.
          const int v1p = e1.sum()-v0;
          return flip ^ flipped_in(vec(v0,v1p),fv) ^ 1;
        } else if (t2==EF) {
          const auto& ef2 = ef_vertices.flat[v2-n];
          const auto e2 = edges[ef2.edge];
          const auto f1 = faces[ef1.face];
          return flip ^ ef1.flip ^ ef2.flip
               ^ !perturbed_predicate<OrientVEFEF>(Xi(v0),EX(e2),EX(e1),FX(f1));
        } else /* t2==FFF */ {
          const auto j = next_two(fff_vertices[v2-nn].faces,face);
          const auto f1 = faces[ef1.face],
                     g0 = vec(e1.x,e1.y,fv.sum()-e1.sum()),
                     g1 = faces[j.x],
                     g2 = faces[j.y];
          return flip ^ ef1.flip
               ^ perturbed_predicate<OrientBVFFF>(FX(f1),FX(g0),FX(g2),FX(g1));
        }}
      case C(B,B,EF):
      case C(B,B,FFF): {
        const auto &ef0 = ef_vertices.flat[v0-n],
                   &ef1 = ef_vertices.flat[v1-n];
        const auto e0 = edges[ef0.edge];
        if (ef0.edge == ef1.edge) {
          // As above, the result depends solely on the orientation of v0,v1.
          // We use the fact that EF vertices are sorted along their edges.
          return flip ^ flipped_in(e0,fv) ^ (v0<v1);
        } else if (t2==EF) {
          const auto& ef2 = ef_vertices.flat[v2-n];
          const auto e1 = edges[ef1.edge],
                     e2 = edges[ef2.edge];
          const auto f0 = faces[ef0.face],
                     f1 = faces[ef1.face];
          return flip ^ ef0.flip ^ ef1.flip ^ ef2.flip
               ^ perturbed_predicate<OrientEFEFEF>(EX(e2),EX(e0),FX(f0),EX(e1),FX(f1));
        } else /* t2==FFF */ {
          const auto e1 = edges[ef1.edge];
          const int i01 = e1.contains(e0.x) ? e0.x : e0.y,
                    i0 = e0.sum()-i01,
                    i1 = e1.sum()-i01;
          const auto j = next_two(fff_vertices[v2-nn].faces,face);
          const auto f0 = faces[ef0.face],
                     f1 = faces[ef1.face],
                     g0 = vec(i01,i0,i1),
                     g1 = faces[j.x],
                     g2 = faces[j.y];
          return flip ^ ef0.flip ^ flipped_in(e0,fv)
                      ^ ef1.flip ^ flipped_in(e1,fv)
               ^ perturbed_predicate<OrientBBFFF>(FX(f0),FX(f1),FX(g0),FX(g2),FX(g1));
        }}
      case C(V,EF,EF): {
        const auto &ef1 = ef_vertices.flat[v1-n],
                   &ef2 = ef_vertices.flat[v2-n];
        const auto e1 = edges[ef1.edge],
                   e2 = edges[ef2.edge];
        return flip ^ ef1.flip ^ ef2.flip
             ^ perturbed_predicate<OrientVEFEF>(Xi(v0),EX(e1),EX(e2),FX0(f)); }
      case C(B,EF,EF):
      case C(EF,EF,EF): {
        const auto &ef0 = ef_vertices.flat[v0-n],
                   &ef1 = ef_vertices.flat[v1-n],
                   &ef2 = ef_vertices.flat[v2-n];
        const auto e0 = edges[ef0.edge],
                   e1 = edges[ef1.edge],
                   e2 = edges[ef2.edge];
        const auto f0 = faces[ef0.face];
        return flip ^ ef0.flip ^ ef1.flip ^ ef2.flip
             ^ perturbed_predicate<OrientEFEFEF>(EX(e1),EX(e2),FX0(f),EX(e0),FX(f0)); }
      case C(V,EF,FFF): {
        const auto& ef1 = ef_vertices.flat[v1-n];
        const auto e1 = edges[ef1.edge];
        const auto j = next_two(fff_vertices[v2-nn].faces,face);
        const auto f1 = faces[j.x],
                   f2 = faces[j.y];
        return flip ^ ef1.flip
             ^ perturbed_predicate<OrientFFFF>(EX(e1),Xi(v0),FX0(f),FX(f2),FX(f1)); }
      case C(B,EF,FFF):
      case C(EF,EF,FFF): {
        const auto &ef0 = ef_vertices.flat[v0-n],
                   &ef1 = ef_vertices.flat[v1-n];
        const auto e0 = edges[ef0.edge],
                   e1 = edges[ef1.edge];
        const auto j = next_two(fff_vertices[v2-nn].faces,face);
        const auto f0 = faces[ef0.face],
                   f1 = faces[j.x],
                   f2 = faces[j.y];
        return flip ^ ef0.flip ^ ef1.flip
             ^ perturbed_predicate<OrientEFEFFFF>(EX(e1),EX(e0),FX(f0),FX0(f),FX(f2),FX(f1)); }
      case C(EF,FFF,FFF): {
        const auto& ef0 = ef_vertices.flat[v0-n];
        const auto e0 = edges[ef0.edge];
        const auto j1 = next_two(fff_vertices[v1-nn].faces,face),
                   j2 = next_two(fff_vertices[v2-nn].faces,face);
        return flip ^ ef0.flip ^ perturbed_predicate<OrientEFFFFFFF>(EX(e0),FX0(f),FX(faces[j1.x]),FX(faces[j1.y]),
                                                                                   FX(faces[j2.x]),FX(faces[j2.y])); }
      case C(V,FFF,FFF):
      case C(B,FFF,FFF):
      case C(FFF,FFF,FFF): {
        // Since fff-vertices always have ff-edges sticking out on all four sides, whenever this
        // occurs during mesh CSG the triangle must be negatively oriented; otherwise the other
        // pieces would stick through and interfere.  Unfortunately, taking advantage of this fact
        // involves reliance on *exactly* how add_constraint_edge calls triangle_oriented in
        // simple_triangulate.h.  Thus, the following code is simple but a bit fragile.
        return false; }
      default:
        GEODE_UNREACHABLE(format("Unhandled case %d %d %d",t0,t1,t2));
    }
    #undef C
  }

  VertexId construct_segment_intersection(MutableTriangleTopology& mesh, const Line L1, const Line L2) {
    const int face1 = L1.face,
              face2 = L2.face;
    const int n = faces_to_fff.size();
    const int i = faces_to_fff.get_or_insert(vec(face,face1,face2).sorted(),n);
    if (i == n) {
      // Vertex has not yet been created.  Make it.
      const auto f0 = faces[face],
                 f1 = faces[face1],
                 f2 = faces[face2];
      const auto c = perturbed_construct<ConstructFFF>(tolerance,FX(f0),FX(f1),FX(f2));
      // Choose order so that the faces are oriented
      fff_vertices.append(FaceFaceFaceVertex({c.y?vec(face,face1,face2):vec(face,face2,face1),c.x}));
      // In check mode, verify that the ordering is orientation consistent
      if (CHECK) {
        const auto f = fff_vertices.back().faces;
        #define N(f) cross(IV(X[f.y])-IV(X[f.x]),IV(X[f.z])-IV(X[f.x]))
        GEODE_ASSERT(weak_sign(edet(N(faces[f.x]),N(faces[f.y]),N(faces[f.z]))) >= 0);
        #undef N
      }
    }
    // Regardless of whether the vertex has been constructed in other triangles, it must be added to our local mesh.
    const auto u = vertices.append(X.size()+ef_vertices.flat.size()+i);
    GEODE_UNUSED const auto v = mesh.add_vertex();
    assert(u==v);
    return u;
  }
};

template<int up> struct BelowVEF {
  template<class TV> static PredicateType<4,TV> eval(VA(v),EA(e),FA(f)) {
    const auto de = e1-e0;
    const auto n = ecross(f1-f0,f2-f0);
    return edot(n,de)*(e0[up]-v[up])+edot(n,f0-e0)*de[up];
  }
};

#define BELOW_EFEF_HELPER(i,e0,e1,f0,f1,f2) \
  const auto d##i = e1-e0; \
  const auto n##i = ecross(f1-f0,f2-f0); \
  const auto a##i = edot(n##i,f0-e0), \
             b##i = edot(n##i,d##i);
#define BELOW_EFEF(e00,e01,f00,f01,f02, \
                   e10,e11,f10,f11,f12) \
  BELOW_EFEF_HELPER(0,e00,e01,f00,f01,f02) \
  BELOW_EFEF_HELPER(1,e10,e11,f10,f11,f12) \
  return b0*(b1*(e10[up]-e00[up]) + a1*d1[up]) - a0*b1*d0[up];
template<int up> struct BelowEFEF {
  template<class TV> static PredicateType<7,TV> eval(EA(e0),FA(f0),EA(e1),FA(f1)) {
    BELOW_EFEF(e00,e01,f00,f01,f02,
               e10,e11,f10,f11,f12)
  }
};
template<int up> struct BelowEFEFSame {
  template<class TV> static PredicateType<7,TV> eval(EA(e0),EA(e1),FA(f)) {
    BELOW_EFEF(e00,e01,f0,f1,f2,
               e10,e11,f0,f1,f2)
  }
};

// Ordering for sorting interior edge-face vertices
template<int up> struct Below : public State {
  const Vector<P,3> f;

  Below(const State& S, const int face)
    : State(S)
    , f(Xi(faces[face].x),
        Xi(faces[face].y),
        Xi(faces[face].z)) {}

  // If ints are passed in, they always refer to interior edge-face vertices
  bool operator()(const int i0, const int i1) const {
    if (i0 == i1)
      return false;
    const auto &ef0 = ef_vertices.flat[i0],
               &ef1 = ef_vertices.flat[i1];
    assert(ef0.face == ef1.face);
    return FILTER(ef1.p()[up]-ef0.p()[up],
                  below_interior_helper(ef0,ef1));
  }

  bool below_interior_helper(const EdgeFaceVertex& ef0, const EdgeFaceVertex& ef1) const {
    const auto e0 = edges[ef0.edge],
               e1 = edges[ef1.edge];
    return ef0.flip ^ ef1.flip ^ perturbed_predicate<BelowEFEFSame<up>>(EX(e0),EX(e1),FX0(f));
  }
};

// Geometry policy for triangulation within the given face, with the given axis as up
template<int up> struct PolicyUp : public Policy {
  PolicyUp(State& S, const int face, const Vector<int,3> face_edges)
    : Policy(S,face,face_edges) {}

  // VertexIds can be either interior or boundary edge-face vertices
  bool below(const VertexId v0, const VertexId v1) const {
    const int i0 = vertices[v0],
              i1 = vertices[v1];
    const auto x0 = Xi3(i0),
               x1 = Xi3(i1);
    return FILTER(x1[up]-x0[up],
                  below_helper(i0,i1));
  }

  bool below_helper(int v0, int v1) const {
    // Sort simpler vertices first
    const int n = X.size();
    const bool flip = v0 >= n;
    if (flip)
      swap(v0,v1);

    // Case analysis on the different kinds of vertex patterns.
    // Simpler cases first.  Unlike triangle_oriented above, we only need deal with v and ef here.
    #define TYPE(i) (v##i >= n)
    #define C(t0,t1) (3*(t0)+(t1))
    static const int V = 0, EF = 1;
    switch (C(TYPE(0),TYPE(1))) {
      case C(V,V): {
        const int lo = vertices.flat[0],
                  hi = vertices.flat[1];
        return flip ^ (v0==lo || v1==hi || v1!=lo); }
      case C(V,EF): {
        const auto& ef = ef_vertices.flat[v1-n];
        const auto e = edges[ef.edge];
        const auto f = faces[ef.face]; // Note: for a boundary ef vertex, this will not be the ambient face
        return flip ^ ef.flip ^ perturbed_predicate<BelowVEF<up>>(Xi(v0),EX(e),FX(f)); }
      case C(EF,EF): {
        const auto &ef0 = ef_vertices.flat[v0-n],
                   &ef1 = ef_vertices.flat[v1-n];
        const auto e0 = edges[ef0.edge],
                   e1 = edges[ef1.edge];
        const auto f0 = faces[ef0.face],
                   f1 = faces[ef1.face];
        return flip ^ ef0.flip ^ ef1.flip
             ^ perturbed_predicate<BelowEFEF<up>>(EX(e0),FX(f0),EX(e1),FX(f1)); }
      default:
        GEODE_UNREACHABLE(format("Unhandled case %d %d",TYPE(0),TYPE(1)));
    }
    #undef TYPE
    #undef C
  }
};
}

// Find all intersection vertices and edges
static Tuple<Nested<const EdgeFaceVertex>,Array<const FaceFaceEdge>>
intersection_simplices(const SimplexTree<EV,2>& face_tree) {
  const auto X = face_tree.X;
  const TriangleSoup& faces = face_tree.mesh;
  const SegmentSoup& edges = faces.segment_soup();
  GEODE_ASSERT(face_tree.leaf_size==1);

  // Find edge-face intersections
  Nested<EdgeFaceVertex> ef_vertices; // Edge-face intersection vertices
  {
    // Find ef_vertices
    struct {
      const Ref<const SimplexTree<EV,1>> edge_tree;
      const SimplexTree<EV,2>& face_tree;
      const RawArray<const EV> X;
      Array<EdgeFaceVertex> ef_vertices;

      bool cull(const int ne, const int nf) const { return false; }

      void leaf(const int ne, const int nf) {
        const int edge = edge_tree->prims(ne)[0],
                  face = face_tree.prims(nf)[0];
        const auto ev = edge_tree->mesh->elements[edge];
        const auto fv = face_tree.mesh->elements[face];
        if (!(fv.contains(ev.x) || fv.contains(ev.y))) {
          const auto e0 = Xi(ev.x), e1 = Xi(ev.y),
                     f0 = Xi(fv.x), f1 = Xi(fv.y), f2 = Xi(fv.z);
          if (segment_triangle_intersect(e0,e1,f0,f1,f2)) {
            const auto c = perturbed_construct<ConstructEF>(tolerance,e0,e1,f0,f1,f2);
            ef_vertices.append(EdgeFaceVertex(edge,face,!c.y,c.x));
          }
        }
      }
    } helper({new_<SimplexTree<EV,1>>(edges,X,1),face_tree,X});
    double_traverse(*helper.edge_tree,face_tree,helper);

    // Bucket edge face vertices by edge
    Array<int> counts(edges.elements.size());
    for (const auto& ef : helper.ef_vertices)
      counts[ef.edge]++;
    ef_vertices = Nested<EdgeFaceVertex>(counts,uninit);
    for (const auto& ef : helper.ef_vertices)
      ef_vertices(ef.edge,--counts[ef.edge]) = ef;
  }

  // Sort ef_vertices along each edge
  for (const int e : range(edges.elements.size())) {
    const auto e0 = Xi(edges.elements[e].x),
               e1 = Xi(edges.elements[e].y);
    struct {
      RawArray<const Vector<int,3>> faces;
      RawArray<const EV> X;
      const P e0, e1;
      const IV de;

      bool operator()(const EdgeFaceVertex& i0, const EdgeFaceVertex& i1) const {
        if (i0.face == i1.face)
          return false;
        const auto &f0 = faces[i0.face],
                   &f1 = faces[i1.face];
        return FILTER(dot(de,i1.p()-i0.p()),
                      helper(i0.face,f0,i1.face,f1));
      }

      bool helper(const int i0, const Vector<int,3> f0,
                  const int i1, const Vector<int,3> f1) const {
        if (f0.sorted() == f1.sorted())
          throw ValueError(format("mesh_csg: Duplicate faces found: face %d (%d,%d,%d) = %d (%d,%d,%d)",
                                  i0,f0.x,f0.y,f0.z,i1,f1.x,f1.y,f1.z));
        return segment_triangle_intersections_ordered(e0,e1,FX(f0),FX(f1));
      }
    } less({faces.elements,X,e0,e1,iv(e1)-iv(e0)});
    sort(ef_vertices[e],less);
  }

  // Map from original vertices to faces
  const auto incident_faces = faces.incident_elements();

  // We want to find all intersection edges.  Each such edge belongs to exactly two faces
  // and has two endpoints.  The endpoints are either edge-face intersections (collected above)
  // or loop vertices.  We've already found edge-face intersections, and the loop vertices
  // can be determined purely combinatorially:
  //
  //   loop_vertices = { v in V | v in (f0,f1) in F, f0 = (v,e), (e,f1) in intersections }
  //                 = { v in V | (e,f) in intersections, v in f, f' = (e,v) in F }
  //   simple_intersection_edges = { i0,i1 in intersections | i0 and i1 share two faces }
  //   faces((e,f)) = {f} + {faces adjacent to e}

  Array<FaceFaceEdge> ff_edges;
  {
    // Find all loop vertices and associated intersection edges
    for (const int i0 : range(ef_vertices.flat.size())) {
      const auto& ef0 = ef_vertices.flat[i0];
      const int f0 = ef0.face;
      const auto e0 = edges.elements[ef0.edge];
      // Check for loop vertices
      for (const int v : faces.elements[f0])
        for (const auto f1 : incident_faces[v])
          if (f0 != f1) {
            const auto f1n = faces.elements[f1];
            if (f1n.contains_all(e0)) {
              const bool flip = ef0.flip ^ flipped_in(e0,f1n);
              ff_edges.append(FaceFaceEdge({flip?vec(f1,f0):vec(f0,f1),vec(v,X.size()+i0)}));
            }
          }
    }

    // Find all (e,f) - (e,f) intersection edges (those not ending at loop vertices)
    {
      // Group intersection vertices by face-face pairs.
      // Note that each pair of faces can share at most two intersections.
      Hashtable<Vector<int,2>,Vector<int,2>> faces_to_intersections;
      for (const int i : range(ef_vertices.flat.size())) {
        const int e = ef_vertices.flat[i].edge,
                  f = ef_vertices.flat[i].face;
        const auto e_nodes = edges.elements[e];
        for (const int f1 : incident_faces[e_nodes.x])
          if (faces.elements[f1].contains(e_nodes.y)) {
            auto& ints = faces_to_intersections.get_or_insert(vec(f,f1).sorted(),vec(-1,-1));
            assert(ints.y < 0); // We should always have room
            (ints.x < 0 ? ints.x : ints.y) = i;
          }
      }

      // Every pair of intersections with two common faces generates an intersection edge
      for (const auto& it : faces_to_intersections) {
        const auto ff = it.x;
        const auto i = it.y;
        if (i.y >= 0) { // Two shared intersections
          const auto& ef = ef_vertices.flat[i.x];
          const auto f0 = ef.face,
                     f1 = ff.sum()-ef.face;
          const auto e = edges.elements[ef.edge];
          const auto f = faces.elements[f1];
          const bool flip = ef.flip ^ flipped_in(e,f);
          ff_edges.append(FaceFaceEdge({flip?vec(f0,f1):vec(f1,f0),X.size()+i}));
        }
      }
    }
  }

  // Verify that the order of faces in ff_edges is consistent with the vertex order
  if (CHECK)
    for (const auto& ff : ff_edges) {
      const auto e = ff.nodes;
      const auto a = faces.elements[ff.faces.x],
                 b = faces.elements[ff.faces.y];
      GEODE_ASSERT(weak_sign(edet(cross(IV(X[a.y])-IV(X[a.x]),IV(X[a.z])-IV(X[a.x])),
                                  cross(IV(X[b.y])-IV(X[b.x]),IV(X[b.z])-IV(X[b.x])),
                                  Xi2(e.y)-Xi2(e.x))) >= 0);
    }

  // Simplices computed!
  return tuple(ef_vertices.const_(),ff_edges.const_());
}

namespace {
// A union-find structure that keeps track of relative depth.
// Union-find algorithm copied from UnionFind.h to avoid excessive template trickery.
struct DepthUnionFind {
  struct Info {
    int p; // parent for children, -rank-1 for roots
    int d; // depth(parent)-depth(self)
  };
  Array<Info> info;

  int append() {
    return info.append(Info({-1,0}));
  }

  int extend(const int n) {
    const int base = info.size();
    info.extend(constant_map(n,Info({-1,0})));
    return base;
  }

  bool same(const int i, const int j) const {
    return find(i).p==find(j).p;
  }

  // Computes p = root(i), d = depth(p)-depth(i)
  Info find(int i) const {
    // Path halve, keeping relative depth values up to date
    int d = 0;
    for (;;) {
      const auto pi = info[i];
      d += pi.d;
      if (pi.p < 0)
        return Info({i,d});
      const auto ppi = info[pi.p];
      d += ppi.d;
      if (ppi.p < 0)
        return Info({pi.p,d});
      info[i] = Info({ppi.p,pi.d+ppi.d});
      i = ppi.p;
    }
  }

  // Compute depth(j)-depth(i), assuming they're in the same component
  int delta(const int i, const int j) const {
    const auto ri = find(i),
               rj = find(j);
    assert(ri.p == rj.p);
    return ri.d-rj.d;
  }

  // dij = depth(j)-depth(i)
  void merge(const int i, const int j, const int dij) {
    auto ri = find(i),
         rj = find(j);
    if (ri.p == rj.p)
      GEODE_ASSERT(ri.d-rj.d==dij,"Inconsistent depth calculation, input meshes are not topologically closed");
    else {
      if (info[ri.p].p <= info[rj.p].p) { // Make ri the root
        if (info[ri.p].p == info[rj.p].p)
          info[ri.p].p--;
        info[rj.p] = Info({ri.p,ri.d-dij-rj.d});
      } else if (info[ri.p].p > info[rj.p].p) // Make rj the root
        info[ri.p] = Info({rj.p,rj.d+dij-ri.d});
    }
  }
};
}

template<int up> static void
retriangulate_face(State& S, Array<Vector<int,3>>& cut_faces, Array<int> &original_face_index, DepthUnionFind* const union_find,
                   const int face, Vector<int,3> e, RawArray<int> interior,
                   RawArray<const FaceFaceEdge> ff_edges, RawArray<const int> ffs) {
  // Sort vertices in upwards order, keeping track of permutation parity.
  const auto save_e = e;
  const auto X = S.X;
  auto v = S.faces[face];
  bool flip = false;
  #define C(i,j) \
    if (axis_less<up>(Xi(v[j]),Xi(v[i]))) { \
      swap(v[i],v[j]); \
      swap(e[i],e[j]); /* Trust us */ \
      flip ^= 1; \
    }
  GEODE_SORT_NETWORK(3,C,NOP)
  #undef C

  // Sort ef_vertices for this face in upwards order
  sort(interior,Below<up>(S,face));

  // Collect all of our vertices together into a contiguous numbering.  The boundary vertices form
  // two chains: one short from v.x to v.z, one long from v.x to v.y to v.z.
  PolicyUp<up> P(S,face,save_e);
  auto& vertices = P.vertices;
  Range<IdIter<VertexId>> left_range, interior_range, right_range;
  VertexId vy;
  {
    const auto ef01 = S.ef_vertices.range(e.z),
               ef02 = S.ef_vertices.range(e.y),
               ef12 = S.ef_vertices.range(e.x);
    vertices.flat.resize(2+interior.size()+ef02.size()+ef01.size()+1+ef12.size()); // Same order as below
    int n = 0;
    vertices.flat[n++] = v.x;
    vertices.flat[n++] = v.z;
    #define RANGE(lo,hi) \
      range(IdIter<VertexId>(VertexId(lo)), \
            IdIter<VertexId>(VertexId(hi)))
    #define EXTEND(array,rev) { \
      const auto slice = vertices.flat.slice(n,n+array.size()); \
      slice = X.size() + array; \
      if (rev) \
        std::reverse(slice.begin(),slice.end()); \
      n += array.size(); }
    // Interior
    const int ni = n;
    EXTEND(interior,false)
    interior_range = RANGE(ni,n);
    // Short border chain
    const int nl = n;
    EXTEND(ef02,S.edges[e.y].x!=v.x)
    left_range = RANGE(nl,n);
    // Long border chain
    const int nr = n;
    EXTEND(ef01,S.edges[e.z].x!=v.x)
    vy.id = n;
    vertices.flat[n++] = v.y;
    EXTEND(ef12,S.edges[e.x].x!=v.y)
    right_range = RANGE(nr,n);
    #undef EXTEND
    #undef RANGE
    assert(vertices.size()==n);
  }
  if (flip)
    swap(left_range,right_range);

  // Invert vertices
  Hashtable<int,VertexId> inv_vertices;
  for (const int i : range(vertices.size())) {
    const VertexId v(i);
    inv_vertices.set(vertices[v],v);
  }

  // Decompose into two monotone polygons and triangulate.  At this stage, we're ignoring
  // face-face edges and face-face-face vertices, similar to the separation between point
  // triangulation and constrained triangulation in Delaunay.
  const auto mesh = new_<MutableTriangleTopology>();
  mesh->add_vertices(vertices.size());
  const VertexId lo(0), hi(1);
  Array<VertexId> stack;
  if (!interior.size()) {
    triangulate_monotone_polygon(P,mesh,lo,hi,left_range,right_range,stack);
  } else {
    triangulate_monotone_polygon(P,mesh,lo,hi,left_range,interior_range,stack);
    triangulate_monotone_polygon(P,mesh,lo,hi,interior_range,right_range,stack);
  }

  // Insert all constraint face-face edges, creating face-face-face vertices in the process.
  for (const int i : range(ffs.size())) {
    const int ffi = ffs[int(random_permute(ffs.size(),key+face,i))];
    const auto ff = ff_edges[ffi];
    add_constraint_edge<Policy>(P,mesh,P.constrained,
                                inv_vertices.get(ff.nodes.x),
                                inv_vertices.get(ff.nodes.y),
                                Line({ff.faces.sum()-face,ff.faces.x!=face?ffi:-ffi-1}));
  }

  // Copy mesh into cut_faces
  for (const auto f : mesh->faces()) {
    const auto v = mesh->vertices(f);
    original_face_index.append(face);
    cut_faces.append(vec(vertices[v.x],
                         vertices[v.y],
                         vertices[v.z]));
  }

  if (union_find) {
    // Absorb depth information at the start of all three original edges
    const int base = union_find->extend(mesh->n_faces());
    const auto h = vec(mesh->halfedge(lo),
                       mesh->halfedge(vy),
                       mesh->halfedge(hi));
    const int shift = flip ? 1 : -1;
    for (int i=0;i<3;i++) {
      const int j = (i+shift+3)%3,
                k = (j+shift+3)%3;
      union_find->merge(e[i],base+mesh->face(S.edges[e[i]].x==v[k] ? mesh->left(h[k]) : mesh->reverse(h[j])).id,0);
    }

    // Absorb depth information in the interior of the cut triangle.
    const int ff_base = S.edges.size();
    for (const auto e : mesh->interior_halfedges()) {
      const auto v = mesh->vertices(e);
      if (v.x < v.y) {
        const auto f1 = mesh->face(mesh->reverse(e));
        if (f1.valid()) {
          const auto f0 = mesh->face(e);
          const Line* L = P.constrained.get_pointer(v);
          if (L) {
            const bool flip = L->ff < 0;
            const int ff = flip ? -L->ff-1 : L->ff,
                      start = ff_edges[ff].nodes.x,
                      face2 = ff_edges[ff].faces.x == face ? ff_edges[ff].faces.y : ff_edges[ff].faces.x;
            const int ddepth = S.depth_weight[face2];
            union_find->merge(base+f0.id,base+f1.id,flip?-ddepth:ddepth);
            if (   start==P.vertices[mesh->src(e)]
                || start==P.vertices[mesh->dst(e)])
              union_find->merge(ff_base+ff,base+f0.id,flip?ddepth:0);
          } else {
            union_find->merge(base+f0.id,base+f1.id,0);
          }
        }
      }
    }
  }
}

// Predicates for firing rays along the x axis
namespace {
// Is normal(p0,p1,p2).x > 0?  Equivalently, is tetrahedron p0,p0+inf*x,p1,p2 oriented?
struct OrientedWithX { template<class TV> static inline PredicateType<2,TV> eval(TV p0, TV p1, TV p2) {
  return edet(p1.yz()-p0.yz(),p2.yz()-p0.yz());
}};
static inline bool oriented_with_x(const P p0, const P p1, const P p2) {
  return perturbed_predicate<OrientedWithX>(p0,p1,p2);
}}

// Retriangulate each face w.r.t. the other faces which cut it
static Tuple<Array<const FaceFaceFaceVertex>,Array<Vector<int,3>>,Array<int>>
retriangulate_soup(const SimplexTree<EV,2>& face_tree, Array<const int> depth_weight, DepthUnionFind* const union_find,
                   Nested<const EdgeFaceVertex> ef_vertices, RawArray<const FaceFaceEdge> ff_edges) {
  GEODE_ASSERT(face_tree.leaf_size==1);
  const auto X = face_tree.X;
  const TriangleSoup& faces = face_tree.mesh;

  // Group edge-face vertices by face
  Nested<int> face_to_ef;
  {
    Array<int> counts(faces.elements.size());
    for (const auto& ef : ef_vertices.flat)
      counts[ef.face]++;
    face_to_ef = Nested<int>(counts,uninit);
    for (const int i : range(ef_vertices.flat.size())) {
      const int f = ef_vertices.flat[i].face;
      face_to_ef(f,--counts[f]) = i;
    }
  }

  // Group face-face edges by face
  Nested<int> face_to_ff;
  {
    Array<int> counts(faces.elements.size());
    for (const auto& ff : ff_edges) {
      counts[ff.faces.x]++;
      counts[ff.faces.y]++;
    }
    face_to_ff = Nested<int>(counts,uninit);
    for (const int i : range(ff_edges.size())) {
      const auto f = ff_edges[i].faces;
      face_to_ff(f.x,--counts[f.x]) = i;
      face_to_ff(f.y,--counts[f.y]) = i;
    }
  }

  // Grab edge information
  const SegmentSoup& edges = faces.segment_soup();
  const auto face_edges = faces.triangle_edges();

  // Newly created faces
  Array<Vector<int,3>> cut_faces;
  Array<int> original_face_index;

  // Depth and connectivity information for (1) original edges, (2) ff edges, and (3) cut faces, indexed back to back.
  // The depth of an edge is defined as the depth of its infinitesimal starting section in the cut mesh.
  // In all cases, the depths are for slightly outside the feature in question.  For faces this means a slight
  // motion along the normal, for original edges a slight motion along the edge normal (a vector with positive
  // dot product with both incident triangles), and for ff edges a motion so that we're slightly above both of
  // the intersecting triangles.
  if (union_find) {
    GEODE_ASSERT(!union_find->info.size());
    union_find->extend(edges.elements.size()+ff_edges.size());
  }

  // Retriangulate each face
  Array<FaceFaceFaceVertex> fff_vertices;
  Hashtable<Vector<int,3>,int> faces_to_fff;
  State S(X,ef_vertices,fff_vertices,faces_to_fff,faces.elements,edges.elements,depth_weight);
  for (const int f : range(faces.elements.size())) {
    const auto v = faces.elements[f];

    // Find the three edges bounding this face
    const auto fe = face_edges[f]; // v01,v12,v20
    Vector<int,3> e(fe.y,fe.z,fe.x); // e[3-i-j] connects v[i] and v[j]

    // If the face isn't cut, there's very little to do
    const auto interior = face_to_ef[f];
    if (!interior.size() && !ef_vertices.size(e.x)
                         && !ef_vertices.size(e.y)
                         && !ef_vertices.size(e.z)) {
      original_face_index.append(f);
      cut_faces.append(v);
      if (union_find) {
        const int i = union_find->append();
        union_find->merge(i,e.x,0);
        union_find->merge(i,e.y,0);
        union_find->merge(i,e.z,0);
      }
      continue;
    }

    // Let the longest axis be the upwards sweep axis.  This choice can be made using inexact arithmetic,
    // since it does not affect correctness.
    const int up = bounding_box(X[v.x],X[v.y],X[v.z]).sizes().dominant_axis();
    const auto ffs = face_to_ff[f];
    if (up==0)      retriangulate_face<0>(S,cut_faces,original_face_index,union_find,f,e,interior,ff_edges,ffs);
    else if (up==1) retriangulate_face<1>(S,cut_faces,original_face_index,union_find,f,e,interior,ff_edges,ffs);
    else            retriangulate_face<2>(S,cut_faces,original_face_index,union_find,f,e,interior,ff_edges,ffs);
  }

  // Add one union-find node at infinity, and fire rays until everything is connected to it
  if (union_find) {
    const int infinity = union_find->append();
    const auto incident_faces = faces.incident_elements();
    for (const int f : range(faces.elements.size())) {
      const auto e = face_edges[f];
      if (union_find->same(infinity,e.x))
        continue;
      // Organize the face so that that v0 is the start of edge e.x.
      // We will not use the orientation of v0,v1,v2 in the following, so we don't keep track.
      auto v = faces.elements[f];
      if (edges.elements[e.x].x != v.x)
        swap(v.x,v.y);
      assert(edges.elements[e.x] == v.xy());
      // Let e1,e2,e3 be two infinitesimals, with 1 >> e1 >> e2 >> e3.  We will trace a ray from
      //
      //   q = v0+e1*(v1-v0)+e2*(v2-v0)+e3*normal
      //
      // to infinity along the positive x axis.  Away from an infinitesimal neighborhood
      // of v0, this is equivalent to a ray from v0 to infinity, so any triangle that does
      // not touch v0 can be handled accordingly.  Triangles that touch v0 must be handled
      // specially.  By the choice of q, the depth that we compute will be accurate
      // immediately outside edge v01.  Note that it is *not* necessarily correct anywhere
      // else, since triangle v012 may be cut arbitrarily.
      struct Visitor {
        const SimplexTree<EV,2>& face_tree;
        RawArray<const EV> X;
        const RawArray<const int> depth_weight;
        const P v0,v1,v2;
        const bool orient_v012; // orient_with_x(v0,v1,v2)
        int depth;

        Visitor(const SimplexTree<EV,2>& face_tree, const int face, const Vector<int,3> v, const RawArray<const int> depth_weight)
          : face_tree(face_tree)
          , X(face_tree.X)
          , depth_weight(depth_weight)
          , v0(Xi(v.x))
          , v1(Xi(v.y))
          , v2(Xi(v.z))
          , orient_v012(oriented_with_x(v0,v1,v2))
          , depth(0) {}

        bool cull(const int n) const {
          const auto box = face_tree.boxes[n];
          return                        box.max.x<v0.value().x
                 || v0.value().y<box.min.y || box.max.y<v0.value().y
                 || v0.value().z<box.min.z || box.max.z<v0.value().z;
        }

        void leaf(const int n) {
          const int face_idx = face_tree.prims(n)[0];
          const auto f = face_tree.mesh->elements[face_idx];
          const P p0 = Xi(f.x),
                  p1 = Xi(f.y),
                  p2 = Xi(f.z);
          const bool with_x = oriented_with_x(p0,p1,p2);
          if (!f.contains(v0.seed())) {
            // Triangle doesn't touch v0, so computation is infinitesimal free
            if (   with_x != tetrahedron_oriented(p0,p1,p2,v0)
                && with_x == oriented_with_x(v0,p0,p1)
                && with_x == oriented_with_x(v0,p1,p2)
                && with_x == oriented_with_x(v0,p2,p0))
              goto hit;
          } else if (!f.contains(v1.seed())) {
            // Triangle shares v0 but not v1.  It suffices to consider q = v0+e1*(v1-v0).  The
            // computation is equivalent to firing a ray from v1 -> v1+inf*x against the partially
            // infinite triangle p0+a(p1-p0)+b(p2-p0), {a,b}>=0, as can be seen by scaling around
            // v0 by 1/e1.  This is the same as the no v0 case above except that we do not check
            // against the edge p12, which is now infinitely far away.
            if (   with_x != tetrahedron_oriented(p0,p1,p2,v1)
                && (f.z==v0.seed() || with_x==oriented_with_x(v1,p0,p1))
                && (f.x==v0.seed() || with_x==oriented_with_x(v1,p1,p2))
                && (f.y==v0.seed() || with_x==oriented_with_x(v1,p2,p0)))
              goto hit;
          } else if (!f.contains(v2.seed())) {
            // Triangle shares v0,v1 but not v2.  We must consider the full q = v0+e1*(v1-v0)+e2*(v2-v0).
            // Shift v0 to 0, so that q = e1*v1+e2*v2.
            if (   with_x == (orient_v012 ^ flipped_in(vec(v0.seed(),v1.seed()),f))
                && with_x != tetrahedron_oriented(p0,p1,p2,v2))
              goto hit;
          } else {
            // If f contains v0,v1,v2, we're the start triangle, and we hit iff we're oriented against x.
            if (!with_x)
              goto hit;
          }
          return;
          hit:
          depth += depth_weight[face_idx] * (with_x ? 1 : -1);
        }
      } visitor(face_tree,f,v,depth_weight);
      single_traverse(face_tree,visitor);
      union_find->merge(infinity,e.x,visitor.depth);
    }
  }

  // Done!
  return tuple(fff_vertices.const_(),cut_faces,original_face_index);
}

// The result of split_soup is always "almost" nonmanifold given closed input, but may be slightly
// nonmanifold at particularly complicated loop vertices in the original mesh.  To wit, a loop vertex
// in the original mesh may have more than one one-ring of cut triangles around it in the result.
// If this occurs, we split the loop vertex into multiple coincident copies, one per surrounding one-ring.
static void fix_loops(RawArray<Vector<int,3>> faces, Array<EV>& X, const int n,
                      RawArray<const FaceFaceEdge> ff_edges) {
  // Map potentially bad vertices to a contiguous index
  Hashtable<int> bad;
  {
    // Count how many ff_edges touch each original vertex
    Array<uint8_t> badness(n);
    for (const auto& ff : ff_edges)
      for (const auto v : ff.nodes)
        if (v < n)
          badness[v] = min(2,badness[v]+1);

    // The count must be at least 2 to be potentially bad
    for (const int v : range(n))
      if (badness[v] >= 2)
        bad.set(v);

    // If nothing's bad, we're done
    if (!bad.size())
      return;
  }

  // Collect halfedges incident on bad vertices, and give them a contiguous ordering
  UnionFind union_find;
  Hashtable<Vector<int,2>,int> edges;
  for (const auto f : faces) {
    #define C(i,j,k) \
      if (f.i<n && bad.contains(f.i)) { \
        const int ij = edges.get_or_insert(vec(f.i,f.j),edges.size()), \
                  jk = edges.get_or_insert(vec(f.i,f.k),edges.size()); \
        union_find.extend(edges.size()-union_find.size()); \
        union_find.merge(ij,jk); \
      }
    C(x,y,z) C(y,z,x) C(z,x,y)
    #undef C
  }

  // Map each component to an old or new vertex
  Hashtable<int> seen; // Have we used the old vertex yet?
  Hashtable<int,int> edge_to_copy; // Map from edges (which are roots) to the appropriate loop vertex copy
  for (const auto i : edges)
    if (union_find.is_root(i.y)) {
      const int v = i.x.x;
      edge_to_copy.set(i.y,seen.set(v) ? v : X.append(X[v]));
    }

  // Rebuild mesh in place
  for (auto& f : faces) {
    const auto g = f;
    #define C(i,j) \
      if (g.i<n && bad.contains(g.i)) \
        f.i = edge_to_copy.get(union_find.find(edges.get(vec(g.i,g.j))));
    C(x,y) C(y,z) C(z,x)
    #undef C
  }
}

Tuple<Ref<const TriangleSoup>,Array<EV>>
exact_split_soup(const TriangleSoup& faces, Array<const EV> X, const int depth) {
  Array<int> depth_weight(faces.elements.size(), uninit);
  depth_weight.fill(1);
  return exact_split_soup(faces, X, depth_weight, depth);
}

Tuple<Ref<const TriangleSoup>,Array<EV>>
exact_split_soup(const TriangleSoup& faces, Array<const EV> X, Array<const int> depth_weight, const int depth) {
  IntervalScope scope;

  // Find ef_vertices and ff_halfedges
  const auto face_tree = new_<SimplexTree<EV,2>>(faces,X,1);
  const auto A = intersection_simplices(face_tree);
  const auto ef_vertices = A.x;
  const auto ff_edges = A.y;

  // Optionally compute depths
  Unique<DepthUnionFind> union_find;
  if (depth != all_depths)
    union_find.reset(new DepthUnionFind);

  // Retriangulate mesh and compute depths
  const auto B = retriangulate_soup(face_tree,depth_weight,union_find.get(),ef_vertices,ff_edges);
  const auto fff_vertices = B.x;
  const auto cut_faces = B.y;
  const auto original_face_index = B.z;

  // If desired, extract cut faces at the right depth
  Array<Vector<int,3>> pruned_faces;
  if (union_find) {
    const int infinity = union_find->info.size()-1;
    const int base = faces.segment_soup()->elements.size()+ff_edges.size();
    for (const int f : range(cut_faces.size())) {
      int fdepth = union_find->delta(infinity,base+f);
      int weight = depth_weight[original_face_index[f]];
      if (depth-weight < fdepth && fdepth <= depth) {
        pruned_faces.append(cut_faces[f]);
      }
    }
  } else
    pruned_faces = cut_faces;

  // Concatenate vertices together
  Array<EV> Xs;
  Xs.preallocate(X.size()+ef_vertices.flat.size()+fff_vertices.size());
  Xs.extend(X);
  for (const auto& v : ef_vertices.flat)
    Xs.append_assuming_enough_space(v.rounded);
  for (const auto& v : fff_vertices)
    Xs.append_assuming_enough_space(v.rounded);

  // In depth pruning mode, we may need a little processing to be manifold at loop vertices
  if (union_find)
    fix_loops(pruned_faces,Xs,X.size(),ff_edges);

  // Done!
  return tuple(new_<const TriangleSoup>(pruned_faces),Xs);
}

Tuple<Ref<const TriangleSoup>,Array<TV>> split_soup(const TriangleSoup& faces, Array<const TV> X, Array<const int> depth_weight, const int depth) {
  const auto quant = quantizer(bounding_box(X));
  const auto S = exact_split_soup(faces,amap(quant,X).copy(),depth_weight,depth);
  return tuple(S.x,amap(quant.inverse,S.y).copy());
}

Tuple<Ref<const TriangleSoup>,Array<TV>> split_soup(const TriangleSoup& faces, Array<const TV> X, const int depth) {
  Array<int> depth_weight(faces.elements.size(), uninit);
  depth_weight.fill(1);
  return split_soup(faces, X, depth_weight, depth);
}

// A random looking polynomial vector field for testing purposes.  Doing this in numpy was terribly slow.
static TV signature(const TV p) {
  static const TV cs[20] = {{0.63579617566858204,0.9803866221230878,-1.1149781390749458},{-1.6911029843181062,0.0076849096251670494,-0.20902591156558492},{-0.32936081722995436,1.0215088816527711,-1.5612465562435749},{-0.45614229334747636,-0.70778970138794417,0.81221475328378245},{0.69508749936195235,0.36830278439721859,-0.023097745289497953},{-0.36041115257507639,0.084618397319454405,-0.62507343653099212},{-0.42001958405510559,0.58110444489126467,0.035872312121989956},{-1.0638801780427223,-1.4966105518400179,-0.46276143102821121},{-0.22713523028165017,-0.51887442706005649,-0.61617899144489152},{-0.01614627380526858,-1.0348875675622369,-2.0864245187665253},{0.34335366817123675,1.1129271600488675,0.030032754961424244},{-0.18700129596135318,0.57715102790126815,0.044064679264981095},{0.38502926178803099,0.93873127293758907,-0.024237498658405344},{0.405772588718322,0.27261261469141018,-1.3784370485864426},{0.033162792967982614,-0.53478654089645028,0.66062198865384403},{0.10747984116039729,0.50678316980726434,0.35782550032895966},{1.3356403638933552,0.01886685799296664,-0.92324588402595387},{-0.4121840452935373,0.25449626619085108,-0.1168890420360859},{-0.24743247723688286,0.6995835397565725,1.8017593723959369},{-2.1202767211585711,0.47120110220149913,0.088232150712609772}};
  const auto x = p.x, y = p.y, z = p.z;
  #define c cs[__COUNTER__]
  return c+z*(c+z*(c+z*c))
          +y*(c+z*(c+z*c)+y*(c+z*c+y*c))
          +x*(c+z*(c+z*c)+y*(c+z*c+y*c)+x*(c+z*c+y*c+x*c));
  static_assert(__COUNTER__==sizeof(cs)/sizeof(TV),"");
  #undef c
}

// Perform a surface integral of a vector field over a mesh using degree-7 quadrature.
// Intended for testing purposes.  Quadrature rule from http://arxiv.org/pdf/1111.3827.pdf.
static double mesh_signature(const TriangleSoup& mesh, RawArray<const TV> X) {
  GEODE_ASSERT(mesh.nodes()<=X.size());
  const auto f = signature;
  double sum = 0;
  for (const auto v : mesh.elements) {
    const auto x0 = X[v.x],
               x1 = X[v.y],
               x2 = X[v.z];
    const auto n = .5*cross(x1-x0,x2-x0);
    #define T1(w0,w12) (  f(w0*x0+w12*(x1+x2)) \
                        + f(w0*x1+w12*(x2+x0)) \
                        + f(w0*x2+w12*(x0+x1)))
    #define T2(w0,w1,w2) (  f(w0*x0+w1*x1+w2*x2) \
                          + f(w0*x0+w1*x2+w2*x1) \
                          + f(w0*x1+w1*x0+w2*x2) \
                          + f(w0*x1+w1*x2+w2*x0) \
                          + f(w0*x2+w1*x0+w2*x1) \
                          + f(w0*x2+w1*x1+w2*x0))
    sum += dot(n,  .1253936074493031 *T1(.5134817203287849 ,.2432591398356075)
                 + .07630633834054171*T2(.05071438430720704,.3186441898475371 ,.6306414258452559)
                 + .02766352460147343*T2(.04572082984632032,.08663663134174900,.8676425388119307));
  }
  return sum;
}

}
using namespace geode;

void wrap_mesh_csg() {
  typedef Tuple<Ref<const TriangleSoup>,Array<Vec3>> (*split_fn)(const TriangleSoup&, Array<const Vector<double,3>>, const int);
  GEODE_OVERLOADED_FUNCTION(split_fn,split_soup)
  typedef Tuple<Ref<const TriangleSoup>,Array<exact::Vec3>> (*exact_split_fn)(const TriangleSoup&, Array<const exact::Vec3>, const int);
  GEODE_OVERLOADED_FUNCTION(exact_split_fn,exact_split_soup)

  typedef Tuple<Ref<const TriangleSoup>,Array<Vec3>> (*split_depth_fn)(const TriangleSoup&, Array<const Vector<double,3>>, Array<const int>, const int);
  GEODE_OVERLOADED_FUNCTION_2(split_depth_fn,"split_soup_with_weight",split_soup)
  typedef Tuple<Ref<const TriangleSoup>,Array<exact::Vec3>> (*exact_split_depth_fn)(const TriangleSoup&, Array<const exact::Vec3>, Array<const int>, const int);
  GEODE_OVERLOADED_FUNCTION_2(exact_split_depth_fn,"exact_split_soup_with_weight",exact_split_soup)

  GEODE_FUNCTION(mesh_signature)
}
