// Hash class for STL containers
#pragma once

#include <other/core/math/hash.h>
namespace other {

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
