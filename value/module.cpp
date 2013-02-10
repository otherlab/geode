#include <other/core/python/wrap.h>
using namespace other;

void wrap_value() {
  OTHER_WRAP(value_base)
  OTHER_WRAP(prop)
  OTHER_WRAP(prop_manager)
  OTHER_WRAP(compute)
  OTHER_WRAP(listen)
  OTHER_WRAP(const_value)
}
