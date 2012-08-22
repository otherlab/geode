// Turn a function-like object into an actual function object, sanitizing member function pointers.
// Used by curry and compose.
#pragma once

#include <other/core/utility/move.h>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/if.hpp>
namespace other {

namespace mpl = boost::mpl;

namespace {

template<class T,class R,class... Args> struct Method {
  typedef typename boost::remove_const<T>::type T_;
  typedef typename mpl::if_<boost::is_const<T>,R(T_::*)(Args...) const,R(T_::*)(Args...)>::type F;

  F f;

  Method(F f)
    : f(f) {}

  template<class... A> OTHER_ALWAYS_INLINE R operator()(T* self, A&&... args) const {
    return (self->*f)(other::move(args)...);
  }

  template<class... A> OTHER_ALWAYS_INLINE R operator()(T& self, A&&... args) const {
    return (self.*f)(other::move(args)...);
  }
};

template<class F> struct SanitizeFunction {
  typedef typename remove_reference<F>::type type;
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

}
}
