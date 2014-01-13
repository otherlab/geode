// Turn a function-like object into an actual function object, sanitizing member function pointers.
// Used by curry and compose.
#pragma once

#include <geode/utility/forward.h>
#include <geode/utility/move.h>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/if.hpp>
namespace geode {

namespace mpl = boost::mpl;

namespace {

template<class F> struct SanitizeFunction {
  typedef typename remove_reference<F>::type type;
};

#ifdef GEODE_VARIADIC

template<class T,class R,class... Args> struct Method {
  typedef typename remove_const<T>::type T_;
  typedef typename mpl::if_<is_const<T>,R(T_::*)(Args...) const,R(T_::*)(Args...)>::type F;

  F f;

  Method(F f)
    : f(f) {}

  template<class... Rest> GEODE_ALWAYS_INLINE R operator()(T* self, Rest&&... args) const {
    return (self->*f)(geode::forward<Rest>(args)...);
  }

  template<class... Rest> GEODE_ALWAYS_INLINE R operator()(T& self, Rest&&... args) const {
    return (self.*f)(geode::forward<Rest>(args)...);
  }
};

template<class R,class... Args> struct SanitizeFunction<R(Args...)> {
  typedef R (*type)(Args...);
};

template<class T,class R,class... Args> struct SanitizeFunction<R(T::*)(Args...)> {
  typedef Method<T,R,Args...> type;
};

template<class T,class R,class... Args> struct SanitizeFunction<R(T::*)(Args...) const> {
  typedef Method<const T,R,Args...> type;
};

#else // Unpleasant nonvariadic versions

template<class T,class R,class A0=void,class A1=void,class A2=void,class A3=void,class A4=void,class A5=void> struct Method;

#define GEODE_SANITIZE(ARGS,RARGS,Args,Argsargs,args) \
  GEODE_SANITIZE_2((,GEODE_REMOVE_PARENS(ARGS)),(template<GEODE_REMOVE_PARENS(RARGS)>),(,GEODE_REMOVE_PARENS(Args)),Args,(,GEODE_REMOVE_PARENS(Argsargs)),args)

#define GEODE_SANITIZE_2(CARGS,TRARGS,CArgs,Args,CArgsargs,args) \
  template<class T,class R GEODE_REMOVE_PARENS(CARGS)> struct Method<T,R GEODE_REMOVE_PARENS(CArgs)> { \
    typedef typename remove_const<T>::type T_; \
    typedef typename mpl::if_<is_const<T>,R(T_::*) Args const,R(T_::*) Args>::type F; \
    typedef R result_type; \
    \
    F f; \
    \
    Method(F f) \
      : f(f) {} \
    \
    GEODE_REMOVE_PARENS(TRARGS) GEODE_ALWAYS_INLINE R operator()(T* self GEODE_REMOVE_PARENS(CArgsargs)) const { \
      return (self->*f) args; \
    } \
    \
    GEODE_REMOVE_PARENS(TRARGS) GEODE_ALWAYS_INLINE R operator()(T& self GEODE_REMOVE_PARENS(CArgsargs)) const { \
      return (self.*f) args; \
    } \
  }; \
  \
  template<class R GEODE_REMOVE_PARENS(CARGS)> struct SanitizeFunction<R Args> { \
    typedef R (*type) Args; \
  }; \
  \
  template<class T,class R GEODE_REMOVE_PARENS(CARGS)> struct SanitizeFunction<R(T::*) Args> { \
    typedef Method<T,R GEODE_REMOVE_PARENS(CArgs)> type; \
  }; \
  \
  template<class T,class R GEODE_REMOVE_PARENS(CARGS)> struct SanitizeFunction<R(T::*) Args const> { \
    typedef Method<const T,R GEODE_REMOVE_PARENS(CArgs)> type; \
  };

GEODE_SANITIZE_2((),(),(),(),(),())
GEODE_SANITIZE((class A0),(class R0),(A0),(R0&& a0),(a0))
GEODE_SANITIZE((class A0,class A1),(class R0,class R1),(A0,A1),(R0&& a0,R1&& a1),(a0,a1))
GEODE_SANITIZE((class A0,class A1,class A2),(class R0,class R1,class R2),(A0,A1,A2),(R0&& a0,R1&& a1,R2&& a2),(a0,a1,a2))
GEODE_SANITIZE((class A0,class A1,class A2,class A3),(class R0,class R1,class R2,class R3),(A0,A1,A2,A3),(R0&& a0,R1&& a1,R2&& a2,R3&& a3),(a0,a1,a2,a3))
GEODE_SANITIZE((class A0,class A1,class A2,class A3,class A4),(class R0,class R1,class R2,class R3,class R4),(A0,A1,A2,A3,A4),(R0&& a0,R1&& a1,R2&& a2,R3&& a3,R4&& a4),(a0,a1,a2,a3,a4))

#undef GEODE_SANITIZE_2
#undef GEODE_SANITIZE

#endif

}
}
