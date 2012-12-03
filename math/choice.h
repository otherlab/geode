//#####################################################################
// Function choice
//#####################################################################
#pragma once

#include <other/core/utility/forward.h>
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

#ifdef OTHER_VARIADIC

template<class T0,class... Rest>                                     static inline T0& choice_helper(mpl::int_<0>, T0& x,                     Rest&...) {return x;}
template<class T0,class T1,class... Rest>                            static inline T1& choice_helper(mpl::int_<1>, T0&, T1& x,                Rest&...) {return x;}
template<class T0,class T1,class T2,class... Rest>                   static inline T2& choice_helper(mpl::int_<2>, T0&, T1&, T2& x,           Rest&...) {return x;}
template<class T0,class T1,class T2,class T3,class... Rest>          static inline T3& choice_helper(mpl::int_<3>, T0&, T1&, T2&, T3& x,      Rest&...) {return x;}
template<class T0,class T1,class T2,class T3,class T4,class... Rest> static inline T4& choice_helper(mpl::int_<4>, T0&, T1&, T2&, T3&, T4& x, Rest&...) {return x;}

template<int i,class... Types> static inline auto choice(const Types&... args)
  -> decltype(choice_helper(mpl::int_<i>(),args...)) {
  return choice_helper(mpl::int_<i>(),args...);
}

#else // Unpleasant nonvariadic versions

#define OTHER_CHOICE(n,ARGS,Args,args) \
  OTHER_CHOICE_HELPER_##n(ARGS,Args) \
  template<int i,OTHER_REMOVE_PARENS(ARGS)> static inline auto choice Args \
    -> decltype(choice_helper(mpl::int_<i>(),OTHER_REMOVE_PARENS(args))) { \
    return choice_helper(mpl::int_<i>(),OTHER_REMOVE_PARENS(args)); \
  }

#define OTHER_CHOICE_HELPER(i,ARGS,Args) \
  template<OTHER_REMOVE_PARENS(ARGS)> static inline T##i& choice_helper(mpl::int_<i>, OTHER_REMOVE_PARENS(Args)) { return x##i; }
#define OTHER_CHOICE_HELPER_1(ARGS,Args) OTHER_CHOICE_HELPER(0,ARGS,Args)
#define OTHER_CHOICE_HELPER_2(ARGS,Args) OTHER_CHOICE_HELPER(1,ARGS,Args) OTHER_CHOICE_HELPER_1(ARGS,Args)
#define OTHER_CHOICE_HELPER_3(ARGS,Args) OTHER_CHOICE_HELPER(2,ARGS,Args) OTHER_CHOICE_HELPER_2(ARGS,Args)
#define OTHER_CHOICE_HELPER_4(ARGS,Args) OTHER_CHOICE_HELPER(3,ARGS,Args) OTHER_CHOICE_HELPER_3(ARGS,Args)
#define OTHER_CHOICE_HELPER_5(ARGS,Args) OTHER_CHOICE_HELPER(4,ARGS,Args) OTHER_CHOICE_HELPER_4(ARGS,Args)

OTHER_CHOICE(1,(class T0),(T0& x0),(x0))
OTHER_CHOICE(2,(class T0,class T1),(T0& x0,T1& x1),(x0,x1))
OTHER_CHOICE(3,(class T0,class T1,class T2),(T0& x0,T1& x1,T2& x2),(x0,x1,x2))
OTHER_CHOICE(4,(class T0,class T1,class T2,class T3),(T0& x0,T1& x1,T2& x2,T3& x3),(x0,x1,x2,x3))
OTHER_CHOICE(5,(class T0,class T1,class T2,class T3,class T4),(T0& x0,T1& x1,T2& x2,T3& x3,T4& x4),(x0,x1,x2,x3,x4))

#undef OTHER_CHOICE_HELPER
#undef OTHER_CHOICE

#endif

}
