//#####################################################################
// Function cube
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline auto cube(const T& a)
  -> decltype(a*a*a) {
  return a*a*a;
}

}
