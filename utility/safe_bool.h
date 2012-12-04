// Safe bool utility types and functions
#pragma once

#include <other/core/utility/config.h>
namespace other {

struct SafeBoolHelper{void F() OTHER_EXPORT;};
typedef void (SafeBoolHelper::*SafeBool)();

template<class T> static inline SafeBool safe_bool(const T& x) {
  return x?&SafeBoolHelper::F:0;
}

}
