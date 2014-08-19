#include <geode/mesh/quadric.h>

namespace geode {

Quadric compute_quadric(TriangleTopology const &mesh, RawField<Vector<real,3>, VertexId> const &X, VertexId v) {
  real total = 0;
  Quadric q;
  for (const auto e : mesh.outgoing(v)) {
    if (!mesh.is_boundary(e)) {
      total += q.add_face(mesh, X, mesh.face(e));
    }
  }

  // Normalize
  if (total) {
    const real inv_total = 1/total;
    q.A *= inv_total;
    q.b *= inv_total;
    q.c *= inv_total;
  }

  return q;
}

}
