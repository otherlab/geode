// An expression template equivalent of numpy's arange
#pragma once

#include <geode/array/ArrayExpression.h>
#include <cassert>
namespace geode {

template<> struct IsArray<ARange>:public mpl::true_{};
template<> struct HasCheapCopy<ARange>:public mpl::true_{};

class ARange : public ArrayExpression<int,ARange> {
public:
  typedef int Element;
private:
  int m;
public:

  explicit ARange(const int m)
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

static inline ARange arange(const int n) {
  return ARange(n);
}

}
