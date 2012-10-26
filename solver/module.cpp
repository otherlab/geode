#include <other/core/python/module.h>

void wrap_solver() {
  OTHER_WRAP(brent)
  OTHER_WRAP(powell)
}
