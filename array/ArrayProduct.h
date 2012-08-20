//#####################################################################
// Class ArrayProduct
//#####################################################################
#pragma once

#include <other/core/array/ArrayExpression.h>
#include <other/core/vector/ArithmeticPolicy.h>
#include <cassert>
namespace other {

template<class TArray1,class TArray2> class ArrayProduct;
template<class TArray1,class TArray2> struct IsArray<ArrayProduct<TArray1,TArray2> >:public mpl::true_{};
template<class TArray1,class TArray2> struct HasCheapCopy<ArrayProduct<TArray1,TArray2> >:public mpl::true_{};

template<class TArray1,class TArray2>
class ArrayProduct : public ArrayExpression<typename Product<typename TArray1::Element,typename TArray2::Element>::type,ArrayProduct<TArray1,TArray2>,TArray1,TArray2> {
  typedef typename TArray1::Element T1;typedef typename TArray2::Element T2;
  typedef typename mpl::if_<HasCheapCopy<TArray1>,const TArray1,const TArray1&>::type TArray1View;
  typedef typename mpl::if_<HasCheapCopy<TArray2>,const TArray2,const TArray2&>::type TArray2View;
  typedef typename Product<T1,T2>::type TProduct;
public:
  typedef TProduct Element;

  TArray1View array1;
  TArray2View array2;

  ArrayProduct(const TArray1& array1,const TArray2& array2)
    : array1(array1), array2(array2)
  {}

  int size() const {
    int size = array1.size();
    assert(size==array2.size());
    return size;
  }

  const TProduct operator[](const int i) const {
    return array1[i]*array2[i];
  }
};

template<class T1,class T2,class TArray1,class TArray2> static inline ArrayProduct<TArray1,TArray2>
operator*(const ArrayBase<T1,TArray1>& array1,const ArrayBase<T2,TArray2>& array2) {
  return ArrayProduct<TArray1,TArray2>(array1.derived(),array2.derived());
}

}
