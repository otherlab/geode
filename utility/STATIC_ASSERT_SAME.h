//#####################################################################
// Macro STATIC_ASSERT_SAME
//#####################################################################
#pragma once

#include <boost/type_traits/is_same.hpp>
namespace other {

#define STATIC_ASSERT_SAME(T1,T2) \
  static_assert(boost::is_same<T1,T2>::value,#T1 " and " #T2 " do not match")

}
