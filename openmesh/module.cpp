//#####################################################################
// Module openmesh
//#####################################################################
#include <other/core/python/module.h>
using namespace other;

void wrap_openmesh() {
#ifdef USE_OPENMESH
  OTHER_WRAP(trimesh)
  OTHER_WRAP(decimate)
#endif
}
