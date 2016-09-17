//#####################################################################
// Macro STATIC_ASSERT_SAME
//#####################################################################
#pragma once

namespace geode {

template<class T0,class T1> struct AssertSame { static const bool value = false; };
template<class T> struct AssertSame<T,T> { static const bool value = true; };

#define STATIC_ASSERT_SAME(T0,T1) \
  static_assert(geode::AssertSame<T0,T1>::value,"Given types do not match")

}
