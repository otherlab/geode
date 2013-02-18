#include <other/core/python/module.h>
using namespace other;

void wrap_exact() {
  OTHER_WRAP(exact_tests)
  OTHER_WRAP(delaunay)
}
