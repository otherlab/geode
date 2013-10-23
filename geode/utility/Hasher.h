// Hash class for STL containers
#pragma once

#include <geode/math/hash.h>
namespace geode {

struct Hasher {
  template<class T> size_t operator()(T const& x) const {
    return hash(x);
  }

  bool operator==(Hasher h) const {
    return true;
  }

  bool operator!=(Hasher h) const {
    return false;
  }
};

}
