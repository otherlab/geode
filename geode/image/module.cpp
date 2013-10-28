//#####################################################################
// Module Images
//#####################################################################
#include <geode/python/wrap.h>
using namespace geode;

void wrap_image() {
  GEODE_WRAP(Image)
  GEODE_WRAP(mov)
  GEODE_WRAP(color_utils)
}
