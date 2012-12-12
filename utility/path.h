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

OTHER_CORE_EXPORT string join(const string& p0, const string& p1);
OTHER_CORE_EXPORT string join(const string& p0, const string& p1, const string& p2);
OTHER_CORE_EXPORT string join(const string& p0, const string& p1, const string& p2, const string& p3);

OTHER_CORE_EXPORT string extension(const string& path);

OTHER_CORE_EXPORT string remove_extension(const string& path);

OTHER_CORE_EXPORT string basename(const string& path);

OTHER_CORE_EXPORT string dirname(const string& path);

}
}
