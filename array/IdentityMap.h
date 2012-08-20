//#####################################################################
// Class IdentityMap
//#####################################################################
#pragma once

#include <other/core/array/ArrayExpression.h>
#include <cassert>
namespace other {

template<> struct IsArray<IdentityMap>:public mpl::true_{};
template<> struct HasCheapCopy<IdentityMap>:public mpl::true_{};

class IdentityMap : public ArrayExpression<int,IdentityMap> {
public:
  typedef int Element;
private:
  int m;
public:

  explicit IdentityMap(const int m)
    : m(m)
  {}

  int size() const {
    return m;
  }

  int operator[](const int i) const {
    assert(unsigned(i)<unsigned(m));
    return i;
  }
};
}
