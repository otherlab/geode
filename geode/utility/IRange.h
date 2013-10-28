// Compile time range of integers
#pragma once

#include <geode/utility/config.h>
namespace geode {

template<int n,class... R> struct IRange : public IRange<n-1,R...,mpl::int_<sizeof...(R)>> {};
template<class... R> struct IRange<0,R...> : public Types<R...> {};

}
