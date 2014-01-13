//#####################################################################
// Class ArrayLeftMultiple
//#####################################################################
#pragma once

#include <geode/array/ArrayExpression.h>
#include <geode/vector/ArithmeticPolicy.h>
#include <geode/utility/type_traits.h>
#include <boost/mpl/assert.hpp>
namespace geode {

template<class T1,class TArray2> class ArrayLeftMultiple;
template<class T1,class TArray2> struct IsArray<ArrayLeftMultiple<T1,TArray2> >:public mpl::true_{};
template<class T1,class TArray2> struct HasCheapCopy<ArrayLeftMultiple<T1,TArray2> >:public mpl::true_{};

template<class T1,class TArray2>
class ArrayLeftMultiple : public ArrayExpression<typename Product<T1,typename TArray2::Element>::type,ArrayLeftMultiple<T1,TArray2>,TArray2> {
  typedef typename TArray2::Element T2;
  typedef typename mpl::if_<HasCheapCopy<T1>,const T1,const T1&>::type T1View;
  typedef typename mpl::if_<HasCheapCopy<TArray2>,const TArray2,const TArray2&>::type TArray2View;
  typedef typename Product<T1,T2>::type TProduct;
public:
  typedef TProduct Element;

  T1View c;
  TArray2View array;

  ArrayLeftMultiple(const T1& c, const TArray2& array)
    : c(c), array(array) {}

  int size() const {
    return array.size();
  }

  const TProduct operator[](const int i) const {
    return c*array[i];
  }
};

template<class T1,class T2,class TArray2> static inline typename mpl::if_<mpl::true_,ArrayLeftMultiple<T1,TArray2>,typename Product<T1,typename TArray2::Element>::type>::type
operator*(const T1& c, const ArrayBase<T2,TArray2>& array) {
  return ArrayLeftMultiple<T1,TArray2>(c,array.derived());
}

template<class T1,class T2,class TArray2> static inline typename mpl::if_<mpl::true_,ArrayLeftMultiple<T1,TArray2>,typename Product<T1,typename TArray2::Element>::type>::type
operator/(const ArrayBase<T2,TArray2>& array, const T1& c) {
  BOOST_MPL_ASSERT((is_floating_point<T1>));
  return ArrayLeftMultiple<T1,TArray2>(1/c,array.derived());
}

template<class T1,class T2,class TArray2> static inline typename mpl::if_<mpl::true_,ArrayLeftMultiple<T1,TArray2>,typename Product<T1,typename TArray2::Element>::type>::type
operator*(const ArrayBase<T2,TArray2>& array, const T1& c) {
  BOOST_MPL_ASSERT((is_floating_point<T1>));
  return ArrayLeftMultiple<T1,TArray2>(c,array.derived());
}

}
