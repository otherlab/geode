#pragma once

#include <other/core/array/Array.h>

namespace other {

template<class F,class A> auto amap(const F& f, const A& a)
  -> decltype(Array<decltype(f(a[0])),A::d>()) {
  
  typedef decltype(Array<decltype(f(a[0])),A::d>()) ret;

  ret b(a.sizes(), false);
  
  for (int i = 0; i < a.flat.size(); ++i) {
    b.flat[i] = f(a.flat[i]);
  }
  
  return b;
}

template<class F,class A> A &amap_inplace(const F& f, A& a) {
  
  for (int i = 0; i < a.flat.size(); ++i) {
    a.flat[i] = f(a.flat[i]);
  }
  
  return a;
}

}