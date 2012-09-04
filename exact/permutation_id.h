// Map a permutation to a unique integer id
#pragma once

#include <cassert>
namespace other {

// Map a permutation to a unique integer id in range(n!), destroying the permutation in the process.
// This function must match the python version in sage/simplicity.
static inline int permutation_id(const int n, int* p) {
  int id = 0;
  for (int i=0;i<n-1;i++) {
    int j = i;
    for (int k=j+1;k<n;k++) {
      assert(p[i]!=p[k]); // In order to simulate simplicity, points must be combinatorially distinct
      if (p[j]>p[k])
        j = k;
    }
    swap(p[i],p[j]);
    id = (n-i)*id+j-i;
  }
  return id;
}

}
