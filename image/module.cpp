//#####################################################################
// Module Images
//#####################################################################
#include <other/core/python/module.h>
using namespace other;

void wrap_image() {
  OTHER_WRAP(Image)
  OTHER_WRAP(mov)
  OTHER_WRAP(color_utils)
}
