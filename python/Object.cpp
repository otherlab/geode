//#####################################################################
// Class Object
//#####################################################################
#include <other/core/python/Object.h>
#include <other/core/python/Class.h>
using namespace other;

OTHER_DEFINE_TYPE(Object)

Object::Object() {}
Object::~Object() {}

void wrap_object() {
  Class<Object>("Object")
    .OTHER_INIT();
}
