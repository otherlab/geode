//#####################################################################
// Function choice
//#####################################################################
#pragma once

#include <boost/mpl/int.hpp>
#include <cassert>
namespace other {

namespace mpl = boost::mpl;

template<class T> static inline T& choice(const int i, T& a, T& b) {
  assert(unsigned(i)<2);
  return i?b:a;
}

template<class T> static inline T& choice(const int i, T& a, T& b, T& c) {
  assert(unsigned(i)<3);
  return i==0?a:i==1?b:c;
}

template<class T0,class... Rest>                                     static inline T0& choice_helper(mpl::int_<0>, T0& x,                     Rest&...) {return x;}
template<class T0,class T1,class... Rest>                            static inline T1& choice_helper(mpl::int_<1>, T0&, T1& x,                Rest&...) {return x;}
template<class T0,class T1,class T2,class... Rest>                   static inline T2& choice_helper(mpl::int_<2>, T0&, T1&, T2& x,           Rest&...) {return x;}
template<class T0,class T1,class T2,class T3,class... Rest>          static inline T3& choice_helper(mpl::int_<3>, T0&, T1&, T2&, T3& x,      Rest&...) {return x;}
template<class T0,class T1,class T2,class T3,class T4,class... Rest> static inline T4& choice_helper(mpl::int_<4>, T0&, T1&, T2&, T3&, T4& x, Rest&...) {return x;}

template<int i,class... Types> static inline auto choice(const Types&... args)
  -> decltype(choice_helper(mpl::int_<i>(),args...)) {
  return choice_helper(mpl::int_<i>(),args...);
}

}
