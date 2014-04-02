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

GEODE_CORE_EXPORT string repr(const float x);
GEODE_CORE_EXPORT string repr(const double x);
GEODE_CORE_EXPORT string repr(const long double x);
GEODE_CORE_EXPORT string repr(const string& s);
GEODE_CORE_EXPORT string repr(const char* s);

// Make repr work on nonconst char arrays
static inline string repr(char s[]) {
  const char* p = s;
  return repr(p);
}

// For testing purposes
string str_repr_test(const string& s);

}
