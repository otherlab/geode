#include <geode/python/wrap.h>
using namespace geode;

void wrap_structure() {
  GEODE_WRAP(heap)
  GEODE_WRAP(ranked_tree)
}
