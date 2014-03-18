#include <geode/python/module.h>
#include <geode/python/wrap.h>
#include <geode/exact/circle_csg.h>
using namespace geode;

void wrap_exact() {
  GEODE_WRAP(exact_exact)
  GEODE_WRAP(perturb)
  GEODE_WRAP(predicates)
  GEODE_WRAP(constructions)
  GEODE_WRAP(delaunay)
  GEODE_WRAP(polygon_csg)
  GEODE_WRAP(circle_csg)
  GEODE_WRAP(circle_offsets)
  typedef void(*void_fn_of_nested_circle_arcs)(Nested<CircleArc>);
  GEODE_OVERLOADED_FUNCTION(void_fn_of_nested_circle_arcs,reverse_arcs)
}
