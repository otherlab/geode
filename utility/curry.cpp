// A simpler version of boost::bind

#include <other/core/utility/curry.h>
namespace other {

namespace {

struct A {};

struct B {};

struct C {};

C f(A,B) {
  return C();
}

struct G {
  C g(A,B) {
    return C();
  }

  C h(A,B) const {
    return C();
  }

  C operator()(A,B) const {
    return C();
  }
};

}

OTHER_UNUSED static void curry_test() {
  G g;
  A a;
  B b;

  // Test free functions
  auto fa = curry(f,a);
  C(fa(b));

  // Test nonconst member functions
  auto Gga = curry(&G::g,&g,a);
  C(Gga(b));

  // Test const member functions
  auto Gha = curry(&G::h,const_cast<const G*>(&g),a);
  C(Gha(b));

  // Test function objects
  auto ga = curry(g,a);
  C(ga(b));
}

}
