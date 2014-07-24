// Version of Python's enumerate for iterators and variadic templates
// enumerate(collection) products (index,element&) tuples.
// Enumerate<T0,T1,...> inherits from Types<ITP<0,T0>,ITP<1,T1>,...>.
#pragma once

#include <geode/utility/forward.h>
#include <geode/utility/range.h>
#include <geode/structure/forward.h>
namespace geode {

namespace {
template<class Iter> struct EnumerateIter {
  typedef decltype(*declval<const Iter&>()) R;

  int i;
  Iter it;

  EnumerateIter(const int i, const Iter& it)
    : i(i), it(it) {}

  bool operator!=(const EnumerateIter& o) const { return it != o.it; }
  void operator++() const { ++i; ++it; }
  void operator--() const { --i; --it; }
  Tuple<int,R> operator*() { return Tuple<int,R>(i,*it); }
};
}

template<class Seq> static inline auto enumerate(const Seq& seq)
  -> Range<EnumerateIter<decltype(seq.begin())>> {
  typedef EnumerateIter<decltype(seq.begin())> EI;
  return Range<EI>(EI(0,seq.begin()),EI(int(seq.size()),seq.end()));
}

// An (index,type) pair
template<int i,class T> struct ITP {
  enum {index = i};
  typedef T type;
};

#ifdef GEODE_VARIADIC

template<class Output,class... Inputs> struct EnumerateLoop;
template<class Output> struct EnumerateLoop<Output> : public Output {};
template<class... Outputs,class T,class... Inputs> struct EnumerateLoop<Types<Outputs...>,T,Inputs...>
  : public EnumerateLoop<Types<Outputs...,ITP<sizeof...(Outputs),T>>,Inputs...>::type {};

template<class... Args> struct Enumerate : public EnumerateLoop<Types<>,Args...>::type {};

#else

template<class A0=void,class A1=void,class A2=void,class A3=void,class A4=void> struct Enumerate : public Types<ITP<0,A0>,ITP<1,A1>,ITP<2,A2>,ITP<3,A3>,ITP<4,A4>> {};
template<class A0,class A1,class A2,class A3> struct Enumerate<A0,A1,A2,A3> : public Types<ITP<0,A0>,ITP<1,A1>,ITP<2,A2>,ITP<3,A3>> {};
template<class A0,class A1,class A2> struct Enumerate<A0,A1,A2> : public Types<ITP<0,A0>,ITP<1,A1>,ITP<2,A2>> {};
template<class A0,class A1> struct Enumerate<A0,A1> : public Types<ITP<0,A0>,ITP<1,A1>> {};
template<class A0> struct Enumerate<A0> : public Types<ITP<0,A0>> {};
template<> struct Enumerate<> : public Types<> {};

#endif

}
