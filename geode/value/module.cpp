#include <geode/python/wrap.h>
using namespace geode;

void wrap_value() {
  GEODE_WRAP(value_base)
  GEODE_WRAP(prop)
  GEODE_WRAP(prop_manager)
  GEODE_WRAP(compute)
  GEODE_WRAP(listen)
  GEODE_WRAP(const_value)
}
