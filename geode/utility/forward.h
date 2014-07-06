//#####################################################################
// Header utility/forward
//#####################################################################
#pragma once

#include <geode/utility/config.h>
namespace geode {

struct Hasher;
template<class T> class Ptr;
template<class T> class Ref;
template<class I,class Enable=void> struct Range;
template<class T> struct is_smart_pointer;

// Convenience struct for marking that function semantics have changed
struct Mark {};

// Convenience utility for hiding a type from use in overload resolution
template<class T> struct Hide {
  typedef T type;
};

// Return the first type given (for use in SFINAE)
#ifdef GEODE_VARIADIC
template<class A0,class... Rest> struct First {
  typedef A0 type;
};
#else
template<class A0,class A1=void,class A2=void,class A3=void,class A4=void,class A5=void,class A6=void,class A7=void,class A8=void,class A9=void> struct First {
  typedef A0 type;
};
#endif

// A list of types
#ifdef GEODE_VARIADIC
template<class... Args> struct Types {
  static const int size = sizeof...(Args);
  typedef Types type;
};
template<class... Args> const int Types<Args...>::size;
#else
template<class A0=void,class A1=void,class A2=void,class A3=void,class A4=void,class A5=void,class A6=void,class A7=void,class A8=void,class A9=void> struct Types {
  typedef Types type;
};
#endif

// Null pointer convenience class
struct Null {
  template<class T> operator T() const {
    static_assert(is_smart_pointer<T>::value,"");
    return T();
  }
};
static const Null null = Null();

// Marker for special uninitialized constructors
struct Uninit {};
static const Uninit uninit = Uninit();

// Expand to nothing
#define GEODE_EMPTY()

// GEODE_REMOVE_PARENS((a,b,c)) = a,b,c
#define GEODE_REMOVE_PARENS_HELPER(...) __VA_ARGS__
#define GEODE_REMOVE_PARENS(arg) GEODE_REMOVE_PARENS_HELPER arg

// Print a type at compile time
#define GEODE_PRINT_TYPE(...) typedef typename geode::Types<__VA_ARGS__>::_print _print;

// Mark a class noncopyable
struct Noncopyable {
  Noncopyable() = default;
  Noncopyable(const Noncopyable&) = delete;
  void operator=(const Noncopyable&) = delete;
};

}
