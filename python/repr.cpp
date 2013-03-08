//#####################################################################
// Function repr
//#####################################################################
#include <other/core/python/repr.h>
#include <other/core/python/from_python.h>
#include <other/core/utility/format.h>
#include <cstdio>
namespace other {

string repr(PyObject& x) {
#ifdef OTHER_PYTHON
  return from_python<string>(steal_ref_check(PyObject_Repr(&x)));
#else
  return format("<object of type %s>",x.ob_type->tp_name);
#endif
}

string repr(PyObject* x) {
  return x ? repr(*x) : "None";
}

string repr(const float x) {
  static char buffer[40];
  sprintf(buffer,"%.9g",x);
  return buffer;
}

string repr(const double x) {
  static char buffer[40];
  sprintf(buffer,"%.17g",x);
  return buffer;
}

string repr(const long double x) {
  static char buffer[40];
  sprintf(buffer,"%.21Lg",x);
  return buffer;
}

}
