//#####################################################################
// Function repr
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <string>
namespace geode {

using std::string;

template<class T> inline string repr(const T& x) {
  return x.repr();
}

GEODE_CORE_EXPORT string repr(PyObject& x);
GEODE_CORE_EXPORT string repr(PyObject* x);
GEODE_CORE_EXPORT string repr(const float x);
GEODE_CORE_EXPORT string repr(const double x);
GEODE_CORE_EXPORT string repr(const long double x);
GEODE_CORE_EXPORT string repr(const string& s);

}
