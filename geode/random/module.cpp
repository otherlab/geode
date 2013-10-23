//#####################################################################
// Module Random
//#####################################################################
#include <geode/python/wrap.h>
using namespace geode;

void wrap_random() {
  GEODE_WRAP(Random)
  GEODE_WRAP(sobol)
  GEODE_WRAP(counter)
  GEODE_WRAP(permute)
}
