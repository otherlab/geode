//#####################################################################
// Stream based convertion from T to str
//#####################################################################
//
// This is similar to boost::lexical_cast, but simpler and detects
// operator<< overloads that exist only in the geode namespace.
//
//#####################################################################
#pragma once

#include <sstream>
#include <geode/utility/stl.h>
#include <geode/python/Ref.h>
namespace geode {

using std::string;

static inline string str() {
  return string();
}

template<class T> string str(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

static inline string str(const string& x) {
  return x;
}

static inline string str(const char* x) {
  return x;
}

}
