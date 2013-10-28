//#####################################################################
// Class Tuple<T0>
//#####################################################################
#pragma once

#include <geode/structure/forward.h>
#include <iostream> // needed to avoid header bug on Mac OS X
#include <geode/math/choice.h>
#include <geode/math/hash.h>
#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
#include <geode/utility/stream.h>
namespace geode {

template<class T0>
class Tuple<T0> {
public:
  T0 x;

  Tuple()
    : x()
  {}

  Tuple(const T0& x)
    : x(x)
  {}

  bool operator==(const Tuple& p) const {
    return x==p.x;
  }

  bool operator!=(const Tuple& p) const {
    return !(*this==p);
  }

  bool operator<(const Tuple& p) const {
    return x<p.x;
  }

  bool operator>(const Tuple& p) const {
    return x>p.x;
  }

  void get(T0& a) const {
    a=x;
  }

  template<int i> T0& get() {
    BOOST_STATIC_ASSERT(i==0);
    return x;
  }

  template<int i> const T0& get() const {
    BOOST_STATIC_ASSERT(i==0);
    return x;
  }
};

template<class T0> static inline std::istream& operator>>(std::istream& input,Tuple<T0>& p) {
  return input>>expect('(')>>p.x>>expect(',')>>expect(')');
}

template<class T0> static inline std::ostream& operator<<(std::ostream& output,const Tuple<T0>& p) {
  return output<<'('<<p.x<<",)";
}

template<class T0> static inline Hash hash_reduce(const Tuple<T0>& key) {
  return Hash(key.x);
}

}
