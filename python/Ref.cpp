#include <other/core/python/Ref.h>
namespace other {

void throw_self_owner_mismatch() {
  throw AssertionError("can't convert Ref/Ptr<T> Ref/Ptr<PyObject>; self is different from owner");
}

#ifdef OTHER_PYTHON
void set_self_owner_mismatch() {
  PyErr_Format(PyExc_AssertionError, "can't convert Ref or Ptr to python; self is different from owner");
}
#endif

}
