// Safe bool utility types and functions
#pragma once

#include <geode/utility/config.h>
namespace geode {

namespace {
template<class T> struct SafeBool {
  struct Helper { void f() {}; };
  typedef void (Helper::*type)();
};
}

template<class T,class X> static inline typename SafeBool<T>::type safe_bool(const X& x) {
  return x?&SafeBool<T>::Helper::f:0;
}

}
