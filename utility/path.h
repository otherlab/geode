// Path convenience functions
#pragma once

// We could use boost::filesystem for this, but it's nice to
// keep core independent of as much as possible.

#include <other/core/utility/config.h>
#include <string>
namespace other {
namespace path {

using std::string;

#ifdef _WIN32
const char sep = '\\';
static inline bool is_sep(char c) {
  return c=='\\' || c=='/';
}
#else
const char sep = '/';
static inline bool is_sep(char c) {
  return c==sep;
}
#endif

string join(const string& p, const string& q) OTHER_EXPORT;

string extension(const string& path) OTHER_EXPORT;

string remove_extension(const string& path) OTHER_EXPORT;

string basename(const string& path) OTHER_EXPORT;

}
}
