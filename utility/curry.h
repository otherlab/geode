// Partially apply a function object.
// This is essentially a simpler version of boost::bind.
#pragma once

#include <other/core/utility/SanitizeFunction.h>
#include <other/core/utility/Enumerate.h>
#include <other/core/utility/move.h>
#include <other/core/structure/Tuple.h>
#include <boost/utility/declval.hpp>
namespace other {

namespace {

template<class F,class... Args> struct Curry {
  typedef typename SanitizeFunction<F>::type S;

  const S f;
  const Tuple<Args...> args;

  template<class... Args_> Curry(const F& f, Args_&&... args)
    : f(f), args(args...) {}

  template<class... Rest> OTHER_ALWAYS_INLINE auto operator()(Rest&&... rest) const
    -> decltype(boost::declval<const S&>()(boost::declval<const Args&>()...,rest...)) {
    return call(Enumerate<Args...>(),other::forward<Rest>(rest)...);
  }

private:
  template<class... Enum,class... Rest> OTHER_ALWAYS_INLINE auto call(Types<Enum...>, Rest&&... rest) const 
    -> decltype(boost::declval<const S&>()(boost::declval<const Args&>()...,rest...)) {
    return f(args.template get<Enum::index>()...,other::forward<Rest>(rest)...);
  }
};

}

template<class F,class... Args> static inline Curry<F,typename remove_reference<Args>::type...> curry(const F& f, const Args&... args) {
  return Curry<F,typename remove_reference<Args>::type...>(f,args...);
}

}
