//#####################################################################
// Header ArithmeticPolicy
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
#include <boost/utility/declval.hpp>
namespace other{

template<class T0,class T1> struct Sum{typedef decltype(boost::declval<T0>()+boost::declval<T1>()) type;};
template<class T0,class T1> struct Product{typedef decltype(boost::declval<T0>()*boost::declval<T1>()) type;};

template<class T> struct Transpose{typedef decltype(boost::declval<T>().tranposed()) type;};
template<class T0,class T1> struct ProductTranspose{typedef decltype(boost::declval<T0>()*boost::declval<T1>().transposed()) type;};
template<class T0,class T1> struct TransposeProduct{typedef decltype(boost::declval<T0>().transposed()*boost::declval<T1>()) type;};

template<class T,int m,int n> struct ProductTranspose<Matrix<T,m,n>,UpperTriangularMatrix<T,n>> { typedef Matrix<T,m,n> type; };
template<class T,int m,int n> struct ProductTranspose<UpperTriangularMatrix<T,m>,Matrix<T,m,n>> { typedef Matrix<T,m,n> type; };

}
