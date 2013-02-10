#include <other/core/python/wrap.h>

void wrap_solver() {
  OTHER_WRAP(brent)
  OTHER_WRAP(powell)
}
