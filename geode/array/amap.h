#pragma once

#include <geode/array/ArrayExpression.h>
#include <geode/utility/type_traits.h>
namespace geode {

template<class F,class A> class ArrayMap;
template<class F,class A> struct IsArray<ArrayMap<F,A>>:public mpl::true_{};
template<class F,class A> struct HasCheapCopy<ArrayMap<F,A>>:public mpl::true_{};

template<class F,class A> class ArrayMap : public ArrayExpression<typename remove_const_reference<decltype(declval<const F&>()(declval<const A&>()[0]))>::type,ArrayMap<F,A>,A> {
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

// Unlike amap for normal arrays, this Nested version of amap always makes a copy
template<class F,class T,bool frozen> static inline auto amap(const F& f, const Nested<T,frozen>& a)
  -> Nested<typename remove_const_reference<decltype(f(a(0,0)))>::type,frozen> {
  return Nested<typename remove_const_reference<decltype(f(a(0,0)))>::type,frozen>(a.offsets,amap(f,a.flat).copy());
}

}
