// relative_error function
#pragma once

#include <geode/math/max.h>
#include <geode/vector/ScalarPolicy.h>
#include <geode/utility/type_traits.h>
namespace geode {

template<class T> typename enable_if<IsScalar<T>,T>::type relative_error(const T& a, const T& b, const T& absolute=1e-30) {
  return abs(a-b)/max(abs(a),abs(b),absolute);
}

template<class TV> typename TV::Scalar relative_error(const TV& a, const TV& b, const typename TV::Scalar absolute=1e-30) {
  return (a-b).maxabs()/max(a.maxabs(),b.maxabs(),absolute);
}

}
