//#####################################################################
// Module openmesh
//#####################################################################
#include <other/core/python/wrap.h>
using namespace other;

static bool openmesh_enabled() {
#ifdef USE_OPENMESH
  return true;
#else
  return false;
#endif
}

void wrap_openmesh() {
  OTHER_FUNCTION(openmesh_enabled)
#ifdef USE_OPENMESH
  OTHER_WRAP(trimesh)
  OTHER_WRAP(decimate)
  OTHER_WRAP(curvature)
#endif
}
