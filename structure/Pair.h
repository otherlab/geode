//#####################################################################
// Class Tuple<T0,T1>
//#####################################################################
#pragma once

#include <other/core/structure/forward.h>
#include <iostream> // needed to avoid header bug on Mac OS X
#include <other/core/math/choice.h>
#include <other/core/math/hash.h>
#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/utility/stream.h>
namespace other {

template<class T0,class T1>
class Tuple<T0,T1> {
public:
  T0 x;
  T1 y;

  Tuple()
    : x(), y() {}

  Tuple(const T0& x, const T1& y)
    : x(x), y(y)
  {}

  bool operator==(const Tuple& p) const {
    return x==p.x && y==p.y;
  }

  bool operator!=(const Tuple& p) const {
    return !(*this==p);
  }

  bool operator<(const Tuple& p) const {
    return x<p.x || (x==p.x && y<p.y);
  }

  bool operator>(const Tuple& p) const {
    return x>p.x || (x==p.x && y>p.y);
  }

  void get(T0& a, T1& b) const {
    a = x;
    b = y;
  }

  template<int i> auto get() -> decltype(choice<i>(x,y)) {
    return choice_helper(mpl::int_<i>(),x,y);
  }

  template<int i> auto get() const -> decltype(choice<i>(x,y)) {
    return choice_helper(mpl::int_<i>(),x,y);
  }
};

template<class T0,class T1> static inline std::istream& operator>>(std::istream& input,Tuple<T0,T1>& p) {
  return input>>expect('(')>>p.x>>expect(',')>>p.y>>expect(')');
}

template<class T0,class T1> static inline std::ostream& operator<<(std::ostream& output,const Tuple<T0,T1>& p) {
  return output<<'('<<p.x<<','<<p.y<<')';
}

template<class T0,class T1> static inline Hash hash_reduce(const Tuple<T0,T1>& key) {
  return Hash(key.x,key.y);
}

}
