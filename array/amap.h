#pragma once

#include <other/core/array/ArrayExpression.h>
#include <other/core/utility/remove_const_reference.h>
#include <boost/utility/declval.hpp>
namespace other {

using boost::declval;
template<class F,class A> class ArrayMap;
template<class F,class A> struct IsArray<ArrayMap<F,A>>:public mpl::true_{};
template<class F,class A> struct HasCheapCopy<ArrayMap<F,A>>:public mpl::true_{};

template<class F,class A> class ArrayMap : public ArrayExpression<typename A::Element,ArrayMap<F,A>,A> {
  typedef decltype(declval<const F&>()(declval<const A&>()[0])) Result;
  typedef typename mpl::if_<HasCheapCopy<A>,const A,const A&>::type AView;
public:
  typedef typename remove_const_reference<Result>::type T;

  const F f;
  AView array;

  explicit ArrayMap(const F& f, const A& array)
    : f(f), array(array) {}

  int size() const {
    return array.size();
  }

  Result operator[](const int i) const {
    return f(array[i]);
  }
};

template<class F,class A> static inline ArrayMap<F,A> amap(const F& f, const A& a) {
  return ArrayMap<F,A>(f,a);
}

}
