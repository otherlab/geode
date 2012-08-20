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

  template<class... Args> OTHER_ALWAYS_INLINE auto operator()(Args&&... args) const
    -> decltype(f(g(args...))) {
    return f(g(args...));
  }
};

}

template<class F,class G> static inline Compose<F,G> compose(const F& f, const G& g) {
  return Compose<F,G>(f,g);
}

}
