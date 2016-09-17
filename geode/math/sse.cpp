// SSE tests

#include <geode/math/sse.h>
#ifdef __SSE__
#include <geode/python/wrap.h>
namespace geode {

static Vector<double,2> sse_pack_unpack(const Vector<double,2> a) {
  return unpack(pack<double>(a.x,a.y));
}

}
using namespace geode;

void wrap_sse() {
  GEODE_FUNCTION(sse_pack_unpack)  
}
#endif
