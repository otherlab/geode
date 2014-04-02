//#####################################################################
// Header view
//#####################################################################
//
// Flatten a vector array into a scalar array or vice versa.
//
//#####################################################################
#pragma once

#include <geode/array/RawArray.h>
#include <geode/vector/ScalarPolicy.h>
namespace geode {

// check if it's safe to use vector space operations on the result of scalar_view<TV>
template<class TV> struct ScalarViewIsVectorSpace:public IsScalarVectorSpace<TV>{};
template<class TV,int d> struct ScalarViewIsVectorSpace<Array<TV,d> >:public IsScalarVectorSpace<TV>{};
template<class TV,int d> struct ScalarViewIsVectorSpace<RawArray<TV,d> >:public IsScalarVectorSpace<TV>{};
template<class TV,int d> struct ScalarViewIsVectorSpace<Subarray<TV,d> >:public IsScalarVectorSpace<TV>{};

template<class TV> typename enable_if<IsScalarBlock<TV>,RawArray<typename ScalarPolicy<TV>::type> >::type
scalar_view(TV& block) {
  typedef typename ScalarPolicy<TV>::type T;
  return RawArray<T>(sizeof(TV)/sizeof(T),reinterpret_cast<T*>(&block));
}

template<class T,bool c> struct ConstIf;
template<class T> struct ConstIf<T,false>{typedef T type;};
template<class T> struct ConstIf<T,true>{typedef const T type;};

template<class TA> typename enable_if<IsContiguousArray<TA>,
  RawArray<typename ConstIf<typename ScalarPolicy<typename TA::Element>::type,TA::is_const>::type> >::type
scalar_view(const TA& array) {
  typedef typename TA::Element TV;
  static_assert(IsScalarBlock<TV>::value,"");
  typedef typename ConstIf<typename ScalarPolicy<TV>::type,TA::is_const>::type T;
  return RawArray<T>(sizeof(TV)/sizeof(T)*array.sizes().product(),reinterpret_cast<T*>(array.data()));
}

template<class TA> typename enable_if<IsShareableArray<TA>,
  Array<typename ConstIf<typename ScalarPolicy<typename TA::Element>::type,TA::is_const>::type> >::type
scalar_view_own(const TA& array) {
  typedef typename TA::Element TV;
  static_assert(IsScalarBlock<TV>::value,"");
  typedef typename ConstIf<typename ScalarPolicy<TV>::type,TA::is_const>::type T;
  return Array<T>(sizeof(TV)/sizeof(T)*array.sizes().product(),reinterpret_cast<T*>(array.data()),array.owner());
}

template<class TV,class TA> typename enable_if<IsContiguousArray<TA>,
  RawArray<typename ConstIf<TV,TA::is_const>::type> >::type
vector_view(const TA& array) {
  typedef typename TA::Element T;
  static_assert(is_same<typename ScalarPolicy<TV>::type,typename ScalarPolicy<TV>::type>::value,"");
  const int r = sizeof(TV)/sizeof(T);
  static_assert(r*sizeof(T)==sizeof(TV),"");
  const int n = array.size()/r;
  GEODE_ASSERT(r*n==array.size());
  typedef typename ConstIf<TV,TA::is_const>::type TR;
  return RawArray<TR>(n,reinterpret_cast<TR*>(array.data()));
}

template<class TV,class TA> typename enable_if<IsShareableArray<TA>,
  Array<typename ConstIf<TV,TA::is_const>::type> >::type
vector_view_own(const TA& array) {
  typedef typename TA::Element T;
  static_assert(is_same<typename ScalarPolicy<TV>::type,typename ScalarPolicy<TV>::type>::value,"");
  const int r = sizeof(TV)/sizeof(T);
  static_assert(r*sizeof(T)==sizeof(TV),"");
  const int n = array.size()/r;
  GEODE_ASSERT(r*n==array.size());
  typedef typename ConstIf<TV,TA::is_const>::type TR;
  return Array<TR>(n,reinterpret_cast<TR*>(array.data()),array.owner());
}

template<int d,class TA> inline RawArray<typename ConstIf<Vector<typename TA::Element,d>,TA::is_const>::type>
vector_view(const TA& array) {
  return vector_view<Vector<typename TA::Element,d> >(array);
}

template<int d,class TA> inline Array<typename ConstIf<Vector<typename TA::Element,d>,TA::is_const>::type>
vector_view_own(const TA& array) {
  return vector_view_own<Vector<typename TA::Element,d> >(array);
}

}
