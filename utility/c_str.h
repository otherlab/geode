//#####################################################################
// Convert either const char* or string to const char*
//#####################################################################
#pragma once

#include <string>
namespace other{

using std::string;

static inline const char* c_str(char* s) {
  return s;
}

static inline const char* c_str(const char* s) {
  return s;
}

static inline const char* c_str(const string& s) {
  return s.c_str();
}

// Work for boost::filesystem::path
template<class T> static inline const char* c_str(const T& s) {
  return c_str(s.string());
}

}
