// Compose two functions into a single function object
#pragma once

#include <geode/utility/SanitizeFunction.h>
namespace geode {

using namespace geode;

namespace {

template<class F,class G> struct Compose {
  typedef typename SanitizeFunction<F>::type SF;
  typedef typename SanitizeFunction<G>::type SG;

  const SF f;
  const SG g;

  Compose(const F& f, const G& g)
    : f(f), g(g) {}

#ifdef GEODE_VARIADIC

  template<class... Args> GEODE_ALWAYS_INLINE auto operator()(Args&&... args) const
    -> decltype(f(g(args...))) {
    return f(g(args...));
  }

#else // Unpleasant nonvariadic versions

  struct Unusable {};

  #define GEODE_COMPOSE_CALL(ARGS,Argsargs,args) \
    GEODE_REMOVE_PARENS(ARGS) GEODE_ALWAYS_INLINE auto operator() Argsargs const \
      -> decltype(f(g args)) { \
      return f(g args); \
    }
  GEODE_COMPOSE_CALL((template<class Unused>),(Unused unused=Unusable()),())
  GEODE_COMPOSE_CALL((template<class A0>),(A0&& a0),(a0))
  GEODE_COMPOSE_CALL((template<class A0,class A1>),(A0&& a0,A1&& a1),(a0,a1))
  GEODE_COMPOSE_CALL((template<class A0,class A1,class A2>),(A0&& a0,A1&& a1,A2&& a2),(a0,a1,a2))
  #undef GEODE_COMPOSE_CALL

#endif
};

}

template<class F,class G> static inline Compose<F,G> compose(const F& f, const G& g) {
  return Compose<F,G>(f,g);
}

}
