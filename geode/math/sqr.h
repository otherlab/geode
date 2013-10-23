//#####################################################################
// Function sqr
//#####################################################################
//
// Finds the square.
//
//#####################################################################
#pragma once

namespace geode {

template<class T> static inline auto sqr(const T& a)
  -> decltype(a*a) {
  return a*a;
}

}
