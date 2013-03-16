//#####################################################################
// Module math
//#####################################################################
#include <other/core/math/integer_log.h>
#include <other/core/python/wrap.h>
using namespace other;

void wrap_math() {
  OTHER_WRAP(uint128)
  OTHER_WRAP(numeric_limits)
  OTHER_FUNCTION_2(integer_log,static_cast<int(*)(uint64_t)>(integer_log));
}
