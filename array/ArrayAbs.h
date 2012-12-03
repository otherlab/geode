//#####################################################################
// Class ArrayAbs
//#####################################################################
#pragma once

#include <other/core/array/ArrayExpression.h>
#include <other/core/vector/ArithmeticPolicy.h>
#include <other/core/utility/HasCheapCopy.h>
#include <boost/mpl/if.hpp>
namespace other {

template<class TArray> class ArrayAbs;
template<class TArray> struct IsArray<ArrayAbs<TArray> >:public mpl::true_{};
template<class TArray> struct HasCheapCopy<ArrayAbs<TArray> >:public mpl::true_{};

template<class TArray>
class ArrayAbs : public ArrayExpression<typename TArray::Element,ArrayAbs<TArray>,TArray> {
  typedef typename TArray::Element T;
  typedef typename mpl::if_<HasCheapCopy<TArray>,const TArray,const TArray&>::type TArrayView;
public:
  typedef T Element;

  TArrayView array;

  explicit ArrayAbs(const TArray& array)
    : array(array) {}

  int size() const {
    return array.size();
  }

  const T operator[](const int i) const {
    return abs(array[i]);
  }
};

template<class T,class TArray> static inline ArrayAbs<TArray> abs(const ArrayBase<T,TArray>& array) {
  return ArrayAbs<TArray>(array.derived());
}

}
