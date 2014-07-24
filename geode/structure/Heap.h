// A flexible binary heap template
//
// Many times, when one wants a heap, one needs to track additional features
// such as an inverse map.  The stl versions don't handle this case; we do.
#pragma once

#include <geode/math/min.h>
#include <geode/math/max.h>
#include <geode/utility/range.h>
namespace geode {

// Usage: struct Heap : public HeapBase<Heap> { ... };
// The heap has the structure of a binary heap, so
//   parent(k) = (k-1)/2
//   children(k) = (2k+1,2k+2)
// first(i,j) is true if the ith entry should be above the jth entry (nonstrict comparison).
template<class Derived> struct HeapBase {
  // Do we have a valid heap?
  bool is_heap() const {
    for (const int i : range(1,max(1,size_())))
      if (!first_((i-1)/2,i))
        return false;
    return true;
  }

  // Organize an invalid heap.  O(n) time.
  void make() {
    for (int r=(size_()+1)/2-1;r>=0;r--)
      move_downward(r);
  }

  // Move a node downward to restore heapness.  O(log n) time.
  int move_downward(int p) {
    const int n = size_();
    for (;;) {
      const int c0 = 2*p+1;
      if (c0 >= n)
        break;
      const int c1 = min(2*p+2,n-1);
      if (first_(p,c0) && first_(p,c1))
        break;
      const int c = first_(c0,c1) ? c0 : c1;
      swap_(p,c);
      p = c;
    }
    return p;
  }

  // Move a node upward to restore heapness.  O(log n) time.
  int move_upward(int c) {
    while (c > 0) {
      const int p = (c-1)/2;
      if (first_(p,c))
        break;
      swap_(p,c);
      c = p;
    }
    return c;
  }

  // Move a node in either direction to restore heapness.  O(log n) time.
  int move_up_or_down(const int k) {
    const int up = move_upward(k);
    return k == up ? move_downward(k) : up;
  }

private:
  // These functions must be defined in the derived class
  int size_() const { return static_cast<const Derived&>(*this).size(); }
  bool first_(const int i, const int j) const { return static_cast<const Derived&>(*this).first(i,j); }
  void swap_(const int i, const int j) { return static_cast<Derived&>(*this).swap(i,j); }
};

}
