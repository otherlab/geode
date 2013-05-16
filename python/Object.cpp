//#####################################################################
// Class Object
//#####################################################################
#include <other/core/python/Object.h>
#include <other/core/python/Class.h>
#include <cassert>
namespace other {

OTHER_DEFINE_TYPE(Object)

Object::Object() {
#ifndef NDEBUG
  //const auto self = (PyObject*)this-1;
  //assert(self->ob_refcnt==1); // Partially check that object was constructed inside new_ or a wrapped constructor
#endif
}

Object::~Object() {}

}
using namespace other;

void wrap_object() {
  Class<Object>("Object")
    .OTHER_INIT();
}
