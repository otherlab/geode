#include <other/core/python/module.h>
using namespace other;

void wrap_exact() {
  OTHER_WRAP(perturb)
  OTHER_WRAP(predicates)
  OTHER_WRAP(delaunay)
}
