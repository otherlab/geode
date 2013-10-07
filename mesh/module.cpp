//#####################################################################
// Module mesh
//#####################################################################
#include <other/core/python/wrap.h>
using namespace other;

void wrap_mesh() {
  OTHER_WRAP(polygon_mesh)
  OTHER_WRAP(segment_mesh)
  OTHER_WRAP(triangle_mesh)
  OTHER_WRAP(triangle_subdivision)
  OTHER_WRAP(halfedge_mesh)
  OTHER_WRAP(corner_mesh)
}
