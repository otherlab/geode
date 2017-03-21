//#####################################################################
// Function small_sort
//#####################################################################
//
// Exchanges the values passed to be in sorted (ascending) order
//
//#####################################################################
#pragma once

#include <geode/math/min.h>
#include <geode/math/max.h>
#include <algorithm>
namespace geode {

using std::swap;

template<class T> static inline typename disable_if<is_fundamental<T> >::type small_sort(T& a,T& b) {
  if(b<a) swap(a,b);
}

template<class T> static inline typename enable_if<is_fundamental<T> >::type small_sort(T& a,T& b) {
  T min_ab = min(a,b),
    max_ab = max(a,b);
  a = min_ab;
  b = max_ab;
}

template<class T, class Fn> static inline typename disable_if<is_fundamental<T> >::type small_sort(T& a, T& b, const Fn& pred) {
  if(pred(b,a)) swap(a,b);
}


template<class T> static inline void small_sort(T& a,T& b,T& c) {
  small_sort(a,b);small_sort(b,c);small_sort(a,b);
}
template<class T, class Fn> static inline void small_sort(T& a,T& b,T& c, const Fn& pred) {
  small_sort(a,b,pred);small_sort(b,c,pred);small_sort(a,b,pred);
}

template<class T> static inline void small_sort(T& a,T& b,T& c,T& d) {
  small_sort(a,b);small_sort(c,d);small_sort(a,c);small_sort(b,d);small_sort(b,c);
}
template<class T, class Fn> static inline void small_sort(T& a,T& b,T& c,T& d, const Fn& pred) {
  small_sort(a,b,pred);small_sort(c,d,pred);small_sort(a,c,pred);small_sort(b,d,pred);small_sort(b,c,pred);
}

}
