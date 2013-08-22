//#####################################################################
// Module Geometry
//#####################################################################
#include <other/core/python/wrap.h>
using namespace other;

void wrap_geometry() {
  OTHER_WRAP(box_vector)
  OTHER_WRAP(polygon)
  OTHER_WRAP(implicit)
  OTHER_WRAP(frame_implicit)
  OTHER_WRAP(analytic_implicit)
  OTHER_WRAP(box_tree)
  OTHER_WRAP(particle_tree)
  OTHER_WRAP(simplex_tree)
  OTHER_WRAP(platonic)
  OTHER_WRAP(thick_shell)
  OTHER_WRAP(bezier)
  OTHER_WRAP(segment)
  OTHER_WRAP(surface_levelset)
}
