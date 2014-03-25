// Hybrid optimal / std sort
#pragma once

#include <geode/array/sort.h>
#include <geode/math/optimal_sort.h>
namespace geode {

// Sort an array with optimal sort for up to 4 elements, falling back to std::sort.
// The cutoff is chosen because the advantage of optimal sort vs standard is
// 48%, 16%, 18% for 2, 3, and 4 elements, then falls off to < 5% for 5-7 elements.
template<class TA,class Less> static inline void fallback_sort(const TA& array, const Less& less) {
  const int n = int(array.size());
  #define L()
  #define C(i,j) { \
    auto &a = array[i], &b = array[j]; \
    if (less(b,a)) swap(a,b); }
  switch (n) {
    case 0: case 1: break;
    case 2: GEODE_SORT_NETWORK(2,C,L); break;
    case 3: GEODE_SORT_NETWORK(3,C,L); break;
    case 4: GEODE_SORT_NETWORK(4,C,L); break;
    default: std::sort(array.begin(),array.end(),less);
  }
  #undef L
  #undef C
}

}
