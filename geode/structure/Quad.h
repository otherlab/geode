//#####################################################################
// Class Tuple<T0,T1,T2,T3>
//#####################################################################
#pragma once

#include <geode/structure/forward.h>
#include <iostream> // needed to avoid header bug on Mac OS X
#include <geode/math/choice.h>
#include <geode/math/hash.h>
#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
namespace geode {

template<class T0,class T1,class T2,class T3>
class Tuple<T0,T1,T2,T3> {
public:
  enum { m = 4 };
  T0 x;T1 y;T2 z;T3 w;

  Tuple()
    : x(T0()), y(T1()), z(T2()), w(T3()) {}

  Tuple(const T0& x, const T1& y, const T2& z, const T3& w)
    : x(x), y(y), z(z), w(w) {}

  bool operator==(const Tuple& t) const {
    return x==t.x && y==t.y && z==t.z && w==t.w;
  }

  bool operator!=(const Tuple& t) const {
    return !(*this==t);
  }

  void get(T0& a,T1& b,T2& c,T3& d) const {
    a=x;b=y;c=z;d=w;
  }

  template<int i> auto get() -> decltype(choice<i>(x,y,z,w)) {
    return choice_helper(mpl::int_<i>(),x,y,z,w);
  }

  template<int i> auto get() const -> decltype(choice<i>(x,y,z,w)) {
    return choice_helper(mpl::int_<i>(),x,y,z,w);
  }
};

template<class T0,class T1,class T2,class T3> static inline Hash hash_reduce(const Tuple<T0,T1,T2,T3>& key) {
  return Hash(key.x,key.y,key.z,key.w);
}

}
