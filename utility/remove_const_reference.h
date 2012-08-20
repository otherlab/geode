#pragma once

#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
namespace other {

using boost::remove_const;
using boost::remove_reference;

template<class T> struct remove_const_reference : public remove_const<typename remove_reference<T>::type>{};

}
