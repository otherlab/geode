// Simple, standalone version of boost::mpl
#pragma once

#include <geode/utility/config.h>
namespace geode {
namespace mpl {

// Booleans
template<bool b> struct bool_ { static const bool value = b; };
typedef bool_<true> true_;
typedef bool_<false> false_;

// Ints
template<int i> struct int_ { static const int value = i; };

// Logic
template<class A0=true_,class A1=true_,class A2=true_,class A3=true_,class A4=true_,class A5=true_>
struct and_ : public bool_<A0::value && A1::value && A2::value && A3::value && A4::value && A5::value> {};
template<class A0=false_,class A1=false_,class A2=false_,class A3=false_,class A4=false_,class A5=false_>
struct or_ : public bool_<A0::value || A1::value || A2::value || A3::value || A4::value || A5::value> {};
template<class A> struct not_ : public bool_<(!A::value)> {};

// Conditionals
template<bool c,class A,class B> struct if_c;
template<class A,class B> struct if_c<true,A,B> { typedef A type; };
template<class A,class B> struct if_c<false,A,B> { typedef B type; };
template<class C,class A,class B> struct if_ : public if_c<(C::value?true:false),A,B> {};

}
}
