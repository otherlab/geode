#include <other/core/python/module.h>
using namespace other;

void wrap_exact() {
  OTHER_WRAP(perturb)
  OTHER_WRAP(predicates)
  OTHER_WRAP(constructions)
  OTHER_WRAP(delaunay)
  OTHER_WRAP(polygon_csg)
  OTHER_WRAP(circle_csg)
  OTHER_WRAP(circle_offsets)
}
