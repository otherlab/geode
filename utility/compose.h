// Compose two functions into a single function object
#pragma once

#include <other/core/utility/SanitizeFunction.h>
namespace other {

using namespace other;

namespace {

template<class F,class G> struct Compose {
  typedef typename SanitizeFunction<F>::type SF;
  typedef typename SanitizeFunction<G>::type SG;

  const SF f;
  const SG g;

  Compose(const F& f, const G& g)
    : f(f), g(g) {}

#ifdef OTHER_VARIADIC

  template<class... Args> OTHER_ALWAYS_INLINE auto operator()(Args&&... args) const
    -> decltype(f(g(args...))) {
    return f(g(args...));
  }

#else // Unpleasant nonvariadic versions

  struct Unusable {};

  #define OTHER_COMPOSE_CALL(ARGS,Argsargs,args) \
    OTHER_REMOVE_PARENS(ARGS) OTHER_ALWAYS_INLINE auto operator() Argsargs const \
      -> decltype(f(g args)) { \
      return f(g args); \
    }
  OTHER_COMPOSE_CALL((template<class Unused>),(Unused unused=Unusable()),())
  OTHER_COMPOSE_CALL((template<class A0>),(A0&& a0),(a0))
  OTHER_COMPOSE_CALL((template<class A0,class A1>),(A0&& a0,A1&& a1),(a0,a1))
  OTHER_COMPOSE_CALL((template<class A0,class A1,class A2>),(A0&& a0,A1&& a1,A2&& a2),(a0,a1,a2))
  #undef OTHER_COMPOSE_CALL

#endif
};

}

template<class F,class G> static inline Compose<F,G> compose(const F& f, const G& g) {
  return Compose<F,G>(f,g);
}

}
