//#####################################################################
// Header ArithmeticPolicy
//#####################################################################
#pragma once

#include <geode/vector/forward.h>
#include <geode/utility/type_traits.h>
namespace geode {

template<class T0,class T1> struct Sum{typedef decltype(declval<T0>()+declval<T1>()) type;};
template<class T0,class T1> struct Product{typedef decltype(declval<T0>()*declval<T1>()) type;};

template<class T> struct Transpose{typedef decltype(declval<T>().tranposed()) type;};
template<class T0,class T1> struct ProductTranspose{typedef decltype(declval<T0>()*declval<T1>().transposed()) type;};
template<class T0,class T1> struct TransposeProduct{typedef decltype(declval<T0>().transposed()*declval<T1>()) type;};

template<class T,int m,int n> struct ProductTranspose<Matrix<T,m,n>,UpperTriangularMatrix<T,n>> { typedef Matrix<T,m,n> type; };
template<class T,int m,int n> struct ProductTranspose<UpperTriangularMatrix<T,m>,Matrix<T,m,n>> { typedef Matrix<T,m,n> type; };

}
