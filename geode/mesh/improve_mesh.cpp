#include <geode/mesh/improve_mesh.h>

namespace geode {

bool improve_mesh_inplace(MutableTriangleTopology &mesh, Field<Vector<real,3>, VertexId> const &pos,
                          real min_quality, real max_distance, real min_normal_dot, int max_iter) {
  auto Q = [](Triangle<Vector<real,3>> const &t) { return t.quality(); };
  auto EL = [](VertexId, VertexId) { return false; };
  auto VL = [](VertexId) { return false; };

  return improve_mesh_inplace(mesh, pos, min_quality, max_distance, min_normal_dot, max_iter, EL, VL, Q);
}

// positions are assumed to be at default location
Ref<MutableTriangleTopology> improve_mesh(MutableTriangleTopology const &mesh, real min_quality, real max_distance, real min_normal_dot, int max_iter) {
  FieldId<Vector<real,3>,VertexId> posid(vertex_position_id);
  Ref<MutableTriangleTopology> copy = mesh.copy();
  improve_mesh_inplace(copy, copy->field(posid), min_quality, max_distance, min_normal_dot, max_iter);
  return copy;
}

}

#include <geode/python/wrap.h>

using namespace geode;

void wrap_improve_mesh() {
  GEODE_FUNCTION(improve_mesh);
}
