//#####################################################################
// Module Geometry
//#####################################################################
#include <geode/python/wrap.h>
using namespace geode;

void wrap_geometry() {
  GEODE_WRAP(box_vector)
  GEODE_WRAP(polygon)
  GEODE_WRAP(implicit)
  GEODE_WRAP(frame_implicit)
  GEODE_WRAP(analytic_implicit)
  GEODE_WRAP(box_tree)
  GEODE_WRAP(particle_tree)
  GEODE_WRAP(simplex_tree)
  GEODE_WRAP(platonic)
  GEODE_WRAP(thick_shell)
  GEODE_WRAP(bezier)
  GEODE_WRAP(segment)
  GEODE_WRAP(surface_levelset)
  GEODE_WRAP(offset_mesh)
}
