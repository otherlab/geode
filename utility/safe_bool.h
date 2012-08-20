// Safe bool utility types and functions

#pragma once

namespace other {

struct SafeBoolHelper{void F();};
typedef void (SafeBoolHelper::*SafeBool)();

template<class T> static inline SafeBool safe_bool(const T& x) {
  return x?&SafeBoolHelper::F:0;
}

}
