#pragma once

#include <geode/mesh/TriangleTopology.h>
#include <geode/vector/SymmetricMatrix3x3.h>

namespace geode {

struct Quadric {
  SymmetricMatrix<real,3> A;
  Vector<real,3> b;
  real c;

  Quadric(): c(0) {}

  inline real operator()(Vector<real,3> const &x) {
    real e = dot(x,A*x-b)+c;
    assert(e > -1e-12);
    return max(0,e);
  }

  real add_plane(Vector<real,3> const &n_times_w, Vector<real,3> const &p) {
    real w = n_times_w.magnitude();
    if (w) {
      // u = n/w
      // q(x) = w(u'(x-p))^2
      //      = w(u'x-u'p)^2
      //      = w(x'uu'x-2(pu'u)'x+(u'p)^2)
      //      = x'(nn'/w)x-2(pn'n/w)+(n'p)^2/w
      const real inv_w = 1/w,
              pn = dot(p,n_times_w);
      A += scaled_outer_product(inv_w,n_times_w);
      b += 2*inv_w*pn*n_times_w;
      c += inv_w*sqr(pn); 
    }
    return w;
  }

  real add_face(TriangleTopology const &mesh, RawField<Vector<real,3>, VertexId> const &X, FaceId f) {
    const auto v = mesh.vertices(f);
    const auto p = X[v.x],
               n = cross(X[v.y]-p,X[v.z]-p);
    return add_plane(n, p);
  }
};

class TriangleTopology;
Quadric compute_quadric(TriangleTopology const &mesh, RawField<Vector<real,3>, VertexId> const &X, VertexId v);

}
