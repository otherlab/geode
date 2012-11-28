//#####################################################################
// Class ScalarPolicy
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
#include <other/core/array/forward.h>
#include <boost/mpl/if.hpp>
#include <boost/mpl/not.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/utility/enable_if.hpp>
namespace other {

template<class T> struct IsScalar:public mpl::false_{};
template<class T> struct IsScalar<const T>:public IsScalar<T>{};
template<> struct IsScalar<signed char>:public mpl::true_{};
template<> struct IsScalar<unsigned char>:public mpl::true_{};
template<> struct IsScalar<short>:public mpl::true_{};
template<> struct IsScalar<unsigned short>:public mpl::true_{};
template<> struct IsScalar<int>:public mpl::true_{};
template<> struct IsScalar<unsigned int>:public mpl::true_{};
template<> struct IsScalar<long>:public mpl::true_{};
template<> struct IsScalar<unsigned long>:public mpl::true_{};
template<> struct IsScalar<long long>:public mpl::true_{};
template<> struct IsScalar<unsigned long long>:public mpl::true_{};
template<> struct IsScalar<float>:public mpl::true_{};
template<> struct IsScalar<double>:public mpl::true_{};

template<class T> struct IsScalarBlock:public IsScalar<T>{}; // true if memory layout is contiguous array of scalars
template<class T> struct IsScalarBlock<const T>:public IsScalarBlock<T>{};

template<class T> struct IsScalarVectorSpace:public IsScalar<T>{}; // true if we can compute vector space operations on the underlying array of scalars
template<class T> struct IsScalarVectorSpace<const T>:public IsScalarVectorSpace<T>{};

template<class T,class Enabler=void> struct ScalarPolicy{typedef struct Unusable{} type;};
template<class T> struct ScalarPolicy<const T>{typedef typename boost::add_const<typename ScalarPolicy<T>::type>::type type;};
template<class T> struct ScalarPolicy<T,typename boost::enable_if<mpl::and_<IsScalar<T>,mpl::not_<boost::is_const<T> > > >::type>{typedef T type;};
template<class T> struct ScalarPolicy<T,typename boost::enable_if<typename First<mpl::not_<boost::is_const<T> >,typename T::Scalar>::type>::type>{typedef typename T::Scalar type;};

}
