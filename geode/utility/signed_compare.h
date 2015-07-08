// Utility functions to safely compare two numbers as if they were extended to a common signed type
// signed_less((unsigned int)3,-2) will be false even though ((unsigned int)3 < -2) would be true
#pragma once
#include <geode/utility/config.h>
#include <geode/utility/type_traits.h>

namespace geode {

template<class T0, class T1> auto signed_less(const T0 x0, const T1 x1) -> typename enable_if_c<is_signed<T0>::value == is_signed<T1>::value, bool>::type
{ return x0 < x1; }
template<class SignedT, class UnsignedT> auto signed_less(const SignedT x0, const UnsignedT x1) -> typename enable_if_c<is_signed<SignedT>::value && is_unsigned<UnsignedT>::value, bool>::type
{ return (x0 < 0) || (static_cast<typename make_unsigned<SignedT>::type>(x0) < x1); }
template<class UnsignedT, class SignedT> auto signed_less(const UnsignedT x0, const SignedT x1) -> typename enable_if_c<is_unsigned<UnsignedT>::value && is_signed<SignedT>::value, bool>::type
{ return (0 < x1) && (x0 < static_cast<typename make_unsigned<SignedT>::type>(x1)); }

template<class T0, class T1> auto signed_leq(const T0 x0, const T1 x1) -> typename enable_if_c<is_signed<T0>::value == is_signed<T1>::value, bool>::type
{ return x0 <= x1; }
template<class SignedT, class UnsignedT> auto signed_leq(const SignedT x0, const UnsignedT x1) -> typename enable_if_c<is_signed<SignedT>::value && is_unsigned<UnsignedT>::value, bool>::type
{ return (x0 <= 0) || (static_cast<typename make_unsigned<SignedT>::type>(x0) <= x1); }
template<class UnsignedT, class SignedT> auto signed_leq(const UnsignedT x0, const SignedT x1) -> typename enable_if_c<is_unsigned<UnsignedT>::value && is_signed<SignedT>::value, bool>::type
{ return (0 <= x1) && (x0 <= static_cast<typename make_unsigned<SignedT>::type>(x1)); }

template<class T0, class T1> auto signed_eq(const T0 x0, const T1 x1) -> typename enable_if_c<is_signed<T0>::value == is_signed<T1>::value, bool>::type
{ return x0 == x1; }
template<class SignedT, class UnsignedT> auto signed_eq(const SignedT x0, const UnsignedT x1) -> typename enable_if_c<is_signed<SignedT>::value && is_unsigned<UnsignedT>::value, bool>::type
{ return (x0 >= 0) && (static_cast<typename make_unsigned<SignedT>::type>(x0) == x1); }
template<class UnsignedT, class SignedT> auto signed_eq(const UnsignedT x0, const SignedT x1) -> typename enable_if_c<is_unsigned<UnsignedT>::value && is_signed<SignedT>::value, bool>::type
{ return (0 <= x1) && (x0 == static_cast<typename make_unsigned<SignedT>::type>(x1)); }


} // namespace geode