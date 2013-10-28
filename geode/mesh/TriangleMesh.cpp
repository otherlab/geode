#include <geode/mesh/TriangleMesh.h>
#include <geode/python/Class.h>

namespace other {

GEODE_DEFINE_TYPE(TriangleMesh)

TriangleMesh::TriangleMesh() {
}

TriangleMesh::TriangleMesh(TriangleTopology const &topo, Field<Vector<real,3>, VertexId> const &positions)
: TriangleTopology(topo), positions(positions) {
}

TriangleMesh::TriangleMesh(RawArray<const Vector<int,3>> const &tris, RawArray<const Vector<real,3>> const &X)
: TriangleTopology(tris), positions(Field<Vector<real,3>,VertexId>(X.copy())) {
}

TriangleMesh::TriangleMesh(TriangleSoup const &soup, RawArray<const Vector<real,3>> const &X)
: TriangleTopology(soup), positions(Field<Vector<real,3>,VertexId>(X.copy())) {
}

Ref<TriangleMesh> TriangleMesh::copy() const {
  return new_<TriangleMesh>(*this, positions.copy());
}

// add a vertex at the given position
VertexId TriangleMesh::add_vertex(Vector<real,3> const &x) {
  VertexId newv = TriangleTopology::add_vertex();
  assert(newv.id == positions.size());
  positions.append(x);
  return newv;
}

// add a bunch of vertices
VertexId TriangleMesh::add_vertices(RawArray<const Vector<real,3>> X) {
  VertexId newv = TriangleTopology::add_vertices(X.size());
  assert(newv.id == positions.size());
  positions.extend(X);
  return newv;
}

// add another TriangleMesh (add its vertices and faces). Effectively runs a garbage collection on the added mesh, but not ourselves.
Tuple<Array<int>, Array<int>, Array<int>> TriangleMesh::add(TriangleMesh const &mesh) {
  Tuple<Array<int>, Array<int>, Array<int>> result = TriangleTopology::add(mesh);

  // add the new positions
  for (int i = 0; i < result.x.size(); ++i) {
    if (result.x[i] != -1) {
      assert(result.x[i] == positions.size());
      positions.append(mesh.positions[i]);
    }
  }

  return result;
}

// Permute vertices: vertices v becomes vertex permutation[v]
void TriangleMesh::permute_vertices(RawArray<const int> permutation, bool check=false) {
  TriangleTopology.permute_vertices(permutation, check);
  positions.permute(permutation);
}

// Compact the data structure, removing all erased primitives. Returns a tuple of permutations for
// vertices, faces, and boundary halfedges, such that the old primitive i now has index permutation[i].
Tuple<Array<int>, Array<int>, Array<int>> TriangleMesh::collect_garbage() {
  auto result = TriangleTopology::collect_garbage();
  positions.permute(result.x);
}

}
