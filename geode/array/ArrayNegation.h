//#####################################################################
// Class ArrayNegation
//#####################################################################
#pragma once

#include <geode/array/ArrayExpression.h>
#include <geode/vector/ArithmeticPolicy.h>
namespace geode {

template<class TArray> class ArrayNegation;
template<class TArray> struct IsArray<ArrayNegation<TArray> >:public mpl::true_{};
template<class TArray> struct HasCheapCopy<ArrayNegation<TArray> >:public mpl::true_{};

template<class TArray>
class ArrayNegation : public ArrayExpression<typename TArray::Element,ArrayNegation<TArray>,TArray> {
  typedef typename TArray::Element T;
  typedef typename mpl::if_<HasCheapCopy<TArray>,const TArray,const TArray&>::type TArrayView;
public:
  typedef T Element;

  TArrayView array;

  explicit ArrayNegation(const TArray& array)
    : array(array) {}

  int size() const {
    return array.size();
  }

  const T operator[](const int i) const {
    return -array[i];
  }
};

template<class T,class TArray> static inline ArrayNegation<TArray> operator-(const ArrayBase<T,TArray>& array) {
  return ArrayNegation<TArray>(array.derived());
}

}
