//#####################################################################
// Function repr
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <string>
namespace other {

using std::string;

template<class T> inline string repr(const T& x) {
  return x.repr();
}

OTHER_CORE_EXPORT string repr(PyObject& x);
OTHER_CORE_EXPORT string repr(const float x);
OTHER_CORE_EXPORT string repr(const double x);
OTHER_CORE_EXPORT string repr(const long double x);

}
