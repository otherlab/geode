// Clang doesn't have the C++11 headers by default, so declare move and forward ourselves
#pragma once

#include <other/core/utility/config.h>
#include <boost/type_traits/remove_reference.hpp>
namespace other {

using boost::remove_reference;

// Mark an object as safely moveable
template<class T> typename remove_reference<T>::type&& move(T&& x) {
  return static_cast<typename remove_reference<T>::type&&>(x);
}

// For perfect forwarding
template<class T> T&& forward(typename remove_reference<T>::type& x) noexcept {
  return static_cast<T&&>(x);
}

}
