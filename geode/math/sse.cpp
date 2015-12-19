// SSE tests

#include <geode/math/sse.h>
#include <geode/python/wrap.h>
namespace geode {

static bool geode_sse_enabled() {
#ifdef GEODE_SSE
  return true;
#else
  return false;
#endif
}

#ifdef GEODE_SSE
static Vector<double,2> sse_pack_unpack(const Vector<double,2> a) {
  return unpack(pack<double>(a.x,a.y));
}
#endif

}
using namespace geode;

void wrap_sse() {
  GEODE_FUNCTION(geode_sse_enabled)
#ifdef GEODE_SSE
  GEODE_FUNCTION(sse_pack_unpack)
#endif
}
