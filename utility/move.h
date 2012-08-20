// Clang doesn't have the C++11 headers by default, so declare move ourselves
#pragma once

#include <other/core/utility/config.h>
#include <boost/type_traits/remove_reference.hpp>
namespace other {

using boost::remove_reference;

template<class T> typename remove_reference<T>::type&& move(T&& x) {
  return static_cast<typename remove_reference<T>::type&&>(x);
}

}
