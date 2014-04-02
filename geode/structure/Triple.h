//#####################################################################
// Class Tuple<T0,T1,T2>
//#####################################################################
#pragma once

#include <geode/structure/forward.h>
#include <iostream> // needed to avoid header bug on Mac OS X
#include <geode/math/choice.h>
#include <geode/math/hash.h>
namespace geode {

template<class T0,class T1,class T2>
class Tuple<T0,T1,T2> {
public:
  enum { m = 3 };
  T0 x;T1 y;T2 z;

  Tuple()
    : x(T0()), y(T1()), z(T2()) {}

  Tuple(const T0& x,const T1& y,const T2& z)
    : x(x), y(y), z(z) {}

  bool operator==(const Tuple& t) const {
    return x==t.x && y==t.y && z==t.z;
  }

  bool operator!=(const Tuple& t) const {
    return !(*this==t);
  }

  void get(T0& a,T1& b,T2& c) const {
    a=x;b=y;c=z;
  }

  template<int i> auto get() -> decltype(choice<i>(x,y,z)) {
    return choice_helper(mpl::int_<i>(),x,y,z);
  }

  template<int i> auto get() const -> decltype(choice<i>(x,y,z)) {
    return choice_helper(mpl::int_<i>(),x,y,z);
  }
};

template<class T0,class T1, class T2> static inline std::istream& operator>>(std::istream& input,Tuple<T0,T1,T2>& p) {
  return input>>expect('(')>>p.x>>expect(',')>>p.y>>expect(',')>>p.z>>expect(')');
}

template<class T0,class T1, class T2> static inline std::ostream& operator<<(std::ostream& output,const Tuple<T0,T1,T2>& p) {
  return output<<'('<<p.x<<','<<p.y<<','<<p.z<<')';
}

template<class T0,class T1,class T2> static inline Hash hash_reduce(const Tuple<T0,T1,T2>& key) {
  return Hash(key.x,key.y,key.z);
}

}
