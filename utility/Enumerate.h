// Version of Python's enumerate for variadic templates
// Enumerate<T0,T1,...> inherits from Types<ITP<0,T0>,ITP<1,T1>,...>
#pragma once

#include <other/core/utility/forward.h>
#include <other/core/structure/forward.h>
namespace other {

// An (index,type) pair
template<int i,class T> struct ITP {
  enum {index = i};
  typedef T type;
};

template<class Output,class... Inputs> struct EnumerateLoop;
template<class Output> struct EnumerateLoop<Output> : public Output {};
template<class... Outputs,class T,class... Inputs> struct EnumerateLoop<Types<Outputs...>,T,Inputs...>
  : public EnumerateLoop<Types<Outputs...,ITP<sizeof...(Outputs),T>>,Inputs...>::type {};

template<class... Args> struct Enumerate : public EnumerateLoop<Types<>,Args...>::type {};

}
