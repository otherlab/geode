//#####################################################################
// Class ArrayDifference
//#####################################################################
#pragma once

#include <geode/array/ArrayExpression.h>
#include <geode/vector/ArithmeticPolicy.h>
#include <geode/utility/HasCheapCopy.h>
#include <cassert>
namespace geode {

template<class TArray1,class TArray2> class ArrayDifference;
template<class TArray1,class TArray2> struct IsArray<ArrayDifference<TArray1,TArray2> >:public mpl::true_{};
template<class TArray1,class TArray2> struct HasCheapCopy<ArrayDifference<TArray1,TArray2> >:public mpl::true_{};

template<class TArray1,class TArray2>
class ArrayDifference : public ArrayExpression<typename Sum<typename TArray1::Element,typename TArray2::Element>::type,ArrayDifference<TArray1,TArray2>,TArray1,TArray2> {
  typedef typename TArray1::Element T1;typedef typename TArray2::Element T2;
  typedef typename mpl::if_<HasCheapCopy<TArray1>,const TArray1,const TArray1&>::type TArray1View;
  typedef typename mpl::if_<HasCheapCopy<TArray2>,const TArray2,const TArray2&>::type TArray2View;
  typedef typename Sum<T1,T2>::type TDifference;
public:
  typedef TDifference Element;

  TArray1View array1;
  TArray2View array2;

  ArrayDifference(const TArray1& array1, const TArray2& array2)
    : array1(array1), array2(array2) {}

  int size() const {
    int size = array1.size();
    assert(size==array2.size());
    return size;
  }

  const TDifference operator[](const int i) const {
    return array1[i]-array2[i];
  }
};

template<class T1,class T2,class TArray1,class TArray2> ArrayDifference<TArray1,TArray2> operator-(const ArrayBase<T1,TArray1>& array1,const ArrayBase<T2,TArray2>& array2) {
  return ArrayDifference<TArray1,TArray2>(array1.derived(),array2.derived());
}

}
