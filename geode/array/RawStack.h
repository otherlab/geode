//#####################################################################
// Class RawStack
//#####################################################################
//
// Use a RawArray as a stack.  Since RawArray can't resize, the stack must have been correctly
// preallocated (this will be tested with assert).
//
//#####################################################################
#pragma once

#include <geode/array/RawArray.h>
namespace geode {

template<class T>
class RawStack {
public:
  const RawArray<T> data;
  int n;

  explicit RawStack(RawArray<T> data)
    : data(data), n(0) {}

  int size() const {
    return n;
  }

  T& pop() {
    assert(n);
    return data[--n];
  }

  void push(const T& v) {
    assert(data.valid(n)); 
    data[n++] = v;
  }
};

}
