// Safe bool utility types and functions
#pragma once

#include <geode/utility/config.h>
namespace geode {

struct SafeBoolHelper{ GEODE_CORE_EXPORT void F(); };
typedef void (SafeBoolHelper::*SafeBool)();

template<class T> static inline SafeBool safe_bool(const T& x) {
  return x?&SafeBoolHelper::F:0;
}

}
