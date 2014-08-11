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
};

class TriangleTopology;
Quadric compute_quadric(TriangleTopology const &mesh, RawField<Vector<real,3>, VertexId> const &X, VertexId v);

}
