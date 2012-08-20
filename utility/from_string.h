#pragma once

#include <other/core/utility/const_cast.h>
#include <string>
#include <sstream>
namespace other {

using std::string;

template <class T>
static bool from_string(T& t, const string& s) {
  // for some reason, this does not work on some systems
  std::istringstream iss(s);
  return !(iss >> t).fail();
}

template <>
inline bool from_string(real& t, const string& s) {
  const char* st = s.c_str();
  char* ep;
  t = strtod(st, &ep);
  return st != ep;
}

template <>
inline bool from_string(int& t, const string& s) {
  const char* st = s.c_str();
  char* ep;
  long tt = strtol(st, &ep, 10);
  t = (int)tt;
  return st != ep && t==tt;
}

}
