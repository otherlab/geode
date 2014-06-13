// A expression template reversing an array
#pragma once

#include <geode/array/ArrayExpression.h>
#include <geode/vector/ArithmeticPolicy.h>
namespace geode {

template<class TA> class Reversed;
template<class TA> struct IsArray<Reversed<TA>> : public mpl::true_{};
template<class TA> struct HasCheapCopy<Reversed<TA>> : public mpl::true_{};

template<class TA> class Reversed : public ArrayExpression<typename TA::Element,Reversed<TA>,TA> {
  typedef typename TA::Element T;
  typedef typename mpl::if_<HasCheapCopy<TA>,const TA,const TA&>::type TArrayView;
public:
  typedef T Element;

  TArrayView array;

  explicit Reversed(const TA& array)
    : array(array) {}

  int size() const {
    return array.size();
  }

  auto operator[](const int i) const
    -> decltype(array[i]) {
    return array[array.size()-1-i];
  }
};

template<class T,class TA> static inline Reversed<TA> reversed(const ArrayBase<T,TA>& array) {
  return Reversed<TA>(array.derived());
}

}
