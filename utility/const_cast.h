/// A more convenient const_cast
#pragma once

namespace other {

template<class T> inline T& const_cast_(const T& x) {
  return const_cast<T&>(x);
}

template<class T> inline T* const_cast_(const T* x) {
  return const_cast<T*>(x);
}

}
