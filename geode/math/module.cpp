//#####################################################################
// Module math
//#####################################################################
#include <geode/math/integer_log.h>
#include <geode/math/popcount.h>
#include <geode/python/wrap.h>
using namespace geode;

void wrap_math() {
  GEODE_WRAP(uint128)
  GEODE_WRAP(numeric_limits)
  GEODE_WRAP(optimal_sort)
  GEODE_WRAP(sse)
  GEODE_FUNCTION_2(integer_log,static_cast<int(*)(uint64_t)>(integer_log))
  GEODE_FUNCTION_2(popcount,static_cast<int(*)(uint64_t)>(popcount))
  GEODE_FUNCTION_2(min_bit,static_cast<uint64_t(*)(uint64_t)>(min_bit))
}
