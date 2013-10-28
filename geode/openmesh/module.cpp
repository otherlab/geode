//#####################################################################
// Module openmesh
//#####################################################################
#include <geode/python/wrap.h>
using namespace geode;

static bool openmesh_enabled() {
#ifdef GEODE_OPENMESH
  return true;
#else
  return false;
#endif
}

void wrap_openmesh() {
  GEODE_FUNCTION(openmesh_enabled)
#ifdef GEODE_OPENMESH
  GEODE_WRAP(trimesh)
  GEODE_WRAP(decimate)
  GEODE_WRAP(curvature)
    //  GEODE_WRAP(smooth)
#endif
}
