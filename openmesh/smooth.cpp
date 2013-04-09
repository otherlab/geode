#include <other/core/openmesh/smooth.h>

namespace other {

#if 0

Ref<TriMesh> smooth_mesh(TriMesh const &m, real t) {
  Ref<TriMesh> M = m.copy();
  M->garbage_collection();

  unordered_map<VertexHandle, int> matrix_index;
  unordered_map<int, VertexHandle> mesh_index;

  unordered_map<EdgeHandle, real> edge_weights;

  // make a Laplacian matrix for all vertices that are not locked
  for (auto vh : M->vertex_handles()) {
    // if this vertex has no neighbors, we're done here
    if (M->valence(vh) == 0)
      continue;

    // if this vertex is locked, there's nothing to do
    if (M->status(vh).locked())
      continue;

    // add a row for the vertex and add all necessary vertices for the kernel too

    auto neighbors = M->vertex_one_ring(vh);

    for (auto n : neighbors) {

    }
  }

  return M;
}

#endif

}
