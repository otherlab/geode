// Thickened shells generalizing capsules

#include <geode/geometry/ThickShell.h>
#include <geode/array/view.h>
#include <geode/math/constants.h>
#include <geode/math/copysign.h>
#include <geode/mesh/SegmentSoup.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/python/cast.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
#include <geode/utility/str.h>
#include <geode/vector/normalize.h>
#include <geode/vector/SymmetricMatrix2x2.h>
namespace geode {

typedef double T;
typedef Vector<T,3> TV;
GEODE_DEFINE_TYPE(ThickShell)
using Log::cout;
using std::endl;

ThickShell::ThickShell(const SegmentSoup& mesh, Array<const TV> X, Array<const T> radii)
  : segs(mesh.elements), X(X), radii(radii) {
  GEODE_ASSERT(X.size()==mesh.nodes());
  GEODE_ASSERT(radii.size()==mesh.nodes());
}

ThickShell::ThickShell(const TriangleSoup& mesh, Array<const TV> X, Array<const T> radii)
  : tris(mesh.elements), segs(mesh.segment_soup()->elements), X(X), radii(radii) {
  GEODE_ASSERT(X.size()==mesh.nodes());
  GEODE_ASSERT(radii.size()==mesh.nodes());
}

static Array<const Vector<int,3>> py_tris(Ref<> mesh) {
  if (auto* m = python_cast<TriangleSoup*>(&*mesh))
    return m->elements;
  return Array<const Vector<int,3>>();
}

static Array<const Vector<int,2>> py_segs(Ref<> mesh) {
  if (auto* m = python_cast<TriangleSoup*>(&*mesh))
    return m->segment_soup()->elements;
  else if (auto* m = python_cast<SegmentSoup*>(&*mesh))
    return m->elements;
  else
    throw TypeError(format("ThickShell: expected SegmentSoup or TriangleSoup, got %s",mesh->ob_type->tp_name));
}

ThickShell::ThickShell(Ref<> mesh, Array<const TV> X, Array<const T> radii)
  : tris(py_tris(mesh)), segs(py_segs(mesh)), X(X), radii(radii) {
  const int nodes = max(tris.size()?scalar_view(tris).max()+1:0,segs.size()?scalar_view(segs).max()+1:0);
  GEODE_ASSERT(X.size()==nodes);
  GEODE_ASSERT(radii.size()==nodes);
}

ThickShell::~ThickShell() {}

Tuple<T,TV> ThickShell::phi_normal(const TV& y) const {
  T best_phi = inf;
  TV best_normal;
  const T small = sqrt(numeric_limits<T>::epsilon());
  // Check triangles
  for (const auto& tri : tris) {
    const TV x0 = X[tri.x],
             dx1 = X[tri.y]-x0,
             dx2 = X[tri.z]-x0,
             dy = y-x0;
    const T r0 = radii[tri.x],
            r1 = radii[tri.y],
            r2 = radii[tri.z],
            d11 = sqr_magnitude(dx1),
            d12 = dot(dx1,dx2),
            d22 = sqr_magnitude(dx2);
    const auto dr = vec(r1-r0,r2-r0);
    // We have dot(dxi,normalized(c-dy)) = ri-r0, where c is the closest point.
    // Let z = c-dy, nz = normalized(z), nt the triangle normal, and nz = ai dxi - b nt.  Then
    const SymmetricMatrix<T,2> A(d11,d12,d22);
    const auto a = A.solve_linear_system(dr);
    const T sqr_b = 1-a.x*(a.x*d11+2*a.y*d12)-sqr(a.y)*d22;
    if (sqr_b<0)
      continue;
    const TV n = normalized(cross(dx1,dx2));
    const T ndy = dot(n,dy);
    const T b = copysign(sqrt(sqr_b),ndy);
    // Now we seek k s.t. dy + k nz lies in the triangle plane.  I.e.,
    //   dot(n,dy+k nz) = 0
    //   k = dot(n,dy)/b
    // The radius at the resulting intersection point is given by
    //   r = ei ri = e1 r1 + e2 r2
    //   dy + k nz = ei dxi
    //   dot(dxi,dy) + k(ai |dxi|^2 + aj dot(dxi,dxj)) = ei |dxi|^2 + ej dot(dxi,dxj)
    //   A e = k dr + dot(dy,dxi)
    const T k = ndy/b;
    const auto e = k*a+A.solve_linear_system(vec(dot(dy,dx1),dot(dy,dx2)));
    if (min(e.x,e.y,1-e.x-e.y)<-small)
      continue;
    const T phi = k-(r0+dot(e,dr));
    if (best_phi > phi) {
      best_phi = phi;
      best_normal = b*n-a.x*dx1-a.y*dx2;
    }
  }
  // Check edges.  The formulae are the same as above, but with one fewer i.
  for (const auto& seg : segs) {
    const TV x0 = X[seg.x],
             dx = X[seg.y]-x0,
             dy = y-x0;
    const T r0 = radii[seg.x],
            dr = radii[seg.y]-r0,
            dxx = sqr_magnitude(dx);
    const T a = dr/dxx;
    const T sqr_b = 1-sqr(a)*dxx;
    if (sqr_b<0)
      continue;
    const T b = sqrt(sqr_b);
    // First, compute phi without reconstructing the cylinder normal
    const T dyy = sqr_magnitude(dy),
            dxy = dot(dx,dy),
            ndy = sqrt(max(T(0),dyy-dxy*(dxy/dxx))),
            k = ndy/b,
            e = k*a+dxy/dxx;
    if (min(e,1-e)<-small)
      continue;
    const T phi = k-(r0+e*dr);
    if (best_phi > phi) {
      best_phi = phi;
      // Compute cylinder normal robustly
      const TV u0 = dx.unit_orthogonal_vector(),
               u1 = cross(dx,u0)/sqrt(dxx);
      const auto vu = normalized(vec(dot(dy,u0),dot(dy,u1)));
      best_normal = b*vu.x*u0+b*vu.y*u1-a*dx;
    }
  }
  // Check vertices
  for (const int i : range(X.size())) {
    TV dy = y-X[i];
    const T phi = normalize(dy)-radii[i];
    if (best_phi > phi) {
      best_phi = phi;
      best_normal = dy;
    }
  }
  return tuple(best_phi,best_normal);
}

T ThickShell::phi(const TV& y) const {
  return phi_normal(y).x;
}

TV ThickShell::normal(const TV& y) const {
  return phi_normal(y).y;
}

bool ThickShell::lazy_inside(const TV& y) const {
  return phi_normal(y).x<=0;
}

TV ThickShell::surface(const TV& y) const {
  const auto pn = phi_normal(y);
  return y-pn.x*pn.y;
}

Box<TV> ThickShell::bounding_box() const {
  Box<TV> box;
  for (const int i : range(X.size()))
    box.enlarge(Box<TV>(X[i]).thickened(radii[i]));
  return box;
}

string ThickShell::repr() const {
  return format("ThickShell(TriangleSoup(%s),%s,%s)",str(tris),str(X),str(radii));
}

}
using namespace geode;

void wrap_thick_shell() {
  typedef ThickShell Self;
  Class<Self>("ThickShell")
    .GEODE_INIT(Ref<>,Array<const TV>,Array<const T>)
    ;
}
