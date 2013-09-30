//#####################################################################
// Class ArrayPlusScalar
//#####################################################################
#pragma once

#include <other/core/array/ArrayExpression.h>
#include <other/core/vector/ArithmeticPolicy.h>
#include <other/core/vector/ScalarPolicy.h>
#include <boost/type_traits/is_same.hpp>
namespace other {

template<class T1,class TArray2> class ArrayPlusScalar;
template<class T1,class TArray2> struct IsArray<ArrayPlusScalar<T1,TArray2> >:public mpl::true_{};
template<class T1,class TArray2> struct HasCheapCopy<ArrayPlusScalar<T1,TArray2> >:public mpl::true_{};

template<class T1,class TArray2>
class ArrayPlusScalar : public ArrayExpression<typename Sum<T1,typename TArray2::Element>::type,ArrayPlusScalar<T1,TArray2>,TArray2> {
  typedef typename TArray2::Element T2;
  typedef typename mpl::if_<HasCheapCopy<T1>,const T1,const T1&>::type T1View;
  typedef typename mpl::if_<HasCheapCopy<TArray2>,const TArray2,const TArray2&>::type TArray2View;
  typedef typename Sum<T1,T2>::type TSum;
public:
  typedef TSum Element;

  T1View c;
  TArray2View array;

  ArrayPlusScalar(const T1& c,const TArray2& array)
    : c(c), array(array) {}

  int size() const {
    return array.size();
  }

  const TSum operator[](const int i) const {
    return c+array[i];
  }
};

template<class T1,class T2,class Enable=void> struct ArrayPlusScalarValid:public mpl::false_{};
template<class T1,class T2> struct ArrayPlusScalarValid<T1,T2,typename mpl::if_<mpl::true_,void,typename Sum<T1,T2>::type>::type>:public
  mpl::or_<boost::is_same<typename boost::remove_const<T1>::type,typename boost::remove_const<T2>::type>,IsScalar<T1> >{};

template<class T1,class T2,class TArray2> static inline typename boost::enable_if<ArrayPlusScalarValid<T1,T2>,ArrayPlusScalar<T1,TArray2> >::type
operator+(const T1& c, const ArrayBase<T2,TArray2>& array) {
  return ArrayPlusScalar<T1,TArray2>(c,array.derived());
}

template<class T1,class T2,class TArray2> static inline typename boost::enable_if<ArrayPlusScalarValid<T1,T2>,ArrayPlusScalar<T1,TArray2> >::type
operator+(const ArrayBase<T2,TArray2>& array, const T1& c) {
  return ArrayPlusScalar<T1,TArray2>(c,array.derived());
}

template<class T1,class T2,class TArray2> static inline typename boost::enable_if<ArrayPlusScalarValid<T1,T2>,ArrayPlusScalar<T1,TArray2> >::type
operator-(const ArrayBase<T2,TArray2>& array, const T1& c) {
  return ArrayPlusScalar<T1,TArray2>(-c,array.derived());
}

}
