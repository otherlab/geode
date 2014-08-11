#include <geode/mesh/quadric.h>

namespace geode {

Quadric compute_quadric(TriangleTopology const &mesh, RawField<Vector<real,3>, VertexId> const &X, VertexId v) {
  real total = 0;
  Quadric q;
  for (const auto e : mesh.outgoing(v)) {
    if (!mesh.is_boundary(e)) {
      const auto f = mesh.face(e);
      const auto v = mesh.vertices(f);
      const auto p = X[v.x],
                 n = cross(X[v.y]-p,X[v.z]-p);
      const real w = magnitude(n);
      if (w) {
        total += w;
        // u = n/w
        // q(x) = w(u'(x-p))^2
        //      = w(u'x-u'p)^2
        //      = w(x'uu'x-2(pu'u)'x+(u'p)^2)
        //      = x'(nn'/w)x-2(pn'n/w)+(n'p)^2/w
        const real inv_w = 1/w,
                pn = dot(p,n);
        q.A += scaled_outer_product(inv_w,n);
        q.b += inv_w*pn*n; // We'll multiply by 2 below
        q.c += inv_w*sqr(pn);
      }
    }
  }

  // Normalize
  if (total) {
    const real inv_total = 1/total;
    q.A *= inv_total;
    q.b *= 2*inv_total;
    q.c *= inv_total;
  }

  return q;
}

}
