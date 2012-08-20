//#####################################################################
// Header ArithmeticPolicy
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
#include <boost/utility/declval.hpp>
namespace other{

template<class T1,class T2> struct Sum{typedef decltype(boost::declval<T1>()+boost::declval<T2>()) type;};
template<class T1,class T2> struct Product{typedef decltype(boost::declval<T1>()*boost::declval<T2>()) type;};

template<class T> struct Transpose{typedef decltype(boost::declval<T>().tranposed()) type;};
template<class T1,class T2> struct ProductTranspose{typedef decltype(boost::declval<T1>()*boost::declval<T2>().transposed()) type;};
template<class T1,class T2> struct TransposeProduct{typedef decltype(boost::declval<T1>().transposed()*boost::declval<T2>()) type;};

}
