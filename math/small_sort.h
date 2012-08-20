//#####################################################################
// Function small_sort
//#####################################################################
//
// Exchanges the values passed to be in sorted (ascending) order
//
//#####################################################################
#pragma once

#include <other/core/math/min.h>
#include <other/core/math/max.h>
#include <algorithm>
namespace other {

using std::swap;

template<class T> static inline typename boost::disable_if<boost::is_fundamental<T> >::type small_sort(T& a,T& b) {
  if(b<a) swap(a,b);
}

template<class T> static inline typename boost::enable_if<boost::is_fundamental<T> >::type small_sort(T& a,T& b) {
  T min_ab = min(a,b),
    max_ab = max(a,b);
  a = min_ab;
  b = max_ab;
}

template<class T> static inline void small_sort(T& a,T& b,T& c) {
  small_sort(a,b);small_sort(b,c);small_sort(a,b);
}

template<class T> static inline void small_sort(T& a,T& b,T& c,T& d) {
  small_sort(a,b);small_sort(c,d);small_sort(a,c);small_sort(b,d);small_sort(b,c);
}

}
