//#####################################################################
// Class Tuple<>
//#####################################################################
#pragma once

#include <geode/structure/forward.h>
#include <geode/array/forward.h>
#include <iostream> // needed to avoid header bug on Mac OS X
#include <geode/math/choice.h>
#include <geode/math/hash.h>
#include <geode/utility/stream.h>
namespace geode {

template<>
class Tuple<> {
public:
  bool operator==(const Tuple p) const {
    return true;
  }

  bool operator!=(const Tuple p) const {
    return false;
  }

  bool operator<(const Tuple p) const {
    return false;
  }

  bool operator>(const Tuple p) const {
    return false;
  }

  void get() const {}
};

static inline std::istream& operator>>(std::istream& input,Tuple<> p) {
  return input>>expect('(')>>expect(')');
}

static inline std::ostream& operator<<(std::ostream& output,Tuple<> p) {
  return output<<"()";
}

static inline int hash_reduce(Tuple<> key) {
  return 0;
}

}
