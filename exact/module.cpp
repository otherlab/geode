#include <other/core/python/module.h>
#include <other/core/python/wrap.h>
#include <other/core/exact/circle_csg.h>
using namespace other;


void wrap_exact() {
  OTHER_WRAP(perturb)
  OTHER_WRAP(predicates)
  OTHER_WRAP(constructions)
  OTHER_WRAP(delaunay)
  OTHER_WRAP(polygon_csg)
  OTHER_WRAP(circle_csg)
  OTHER_WRAP(circle_offsets)
  typedef void(*void_fn_of_nested_circle_arcs)(Nested<CircleArc>);
  OTHER_OVERLOADED_FUNCTION(void_fn_of_nested_circle_arcs,reverse_arcs)
}
