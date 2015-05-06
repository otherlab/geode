//#####################################################################
// Module math
//#####################################################################
#include <geode/math/integer_log.h>
#include <geode/math/popcount.h>
#include <geode/python/wrap.h>
#include <cmath>
namespace geode {
// In some configurations, Python.h will attempt to redefine hypot as _hypot
// This helps test that we got a hypot function that works
static real geode_test_hypot(const real a, const real b) {
  using namespace std;
  return hypot(a,b);
}
} // namespace geode

using namespace geode;

void wrap_math() {
  GEODE_WRAP(uint128)
  GEODE_WRAP(numeric_limits)
  GEODE_WRAP(optimal_sort)
  GEODE_FUNCTION(geode_test_hypot)
  GEODE_FUNCTION_2(integer_log,static_cast<int(*)(uint64_t)>(integer_log))
  GEODE_FUNCTION_2(popcount,static_cast<int(*)(uint64_t)>(popcount))
  GEODE_FUNCTION_2(min_bit,static_cast<uint64_t(*)(uint64_t)>(min_bit))
}
