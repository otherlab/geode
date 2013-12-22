//#####################################################################
// Class Object
//#####################################################################
#include <geode/python/Object.h>
#include <geode/python/Class.h>
#include <cassert>
namespace geode {

GEODE_DEFINE_TYPE(Object)

Object::Object() {
#ifndef NDEBUG
  const auto self = (PyObject*)this-1;
  // Partially check that object was constructed inside new_ or a wrapped constructor
  assert(self->ob_refcnt==1 && "Maybe object built outside new_, or compile mixes GEODE_PYTHON and !GEODE_PYTHON");
#endif
}

Object::~Object() {}

}
using namespace geode;

void wrap_object() {
  Class<Object>("Object")
    .GEODE_INIT();
}
