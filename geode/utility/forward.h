//#####################################################################
// Header utility/forward
//#####################################################################
#pragma once

#include <geode/utility/config.h>
namespace geode {

class Owner;
class Object;
struct Buffer;
struct Hasher;
template<class T=Object> class Ptr;
template<class T=Object> class Ref;
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

// Macro to declare new_ as a friend
#ifdef GEODE_VARIADIC

#define GEODE_NEW_FRIEND template<class _T,class... _Args> friend geode::Ref<_T> geode::new_(_Args&&... args);
template<class T,class... Args> static inline Ref<T> new_(Args&&... args);

#else

template<class T> static inline Ref<T> new_();
template<class T,class A0> static inline Ref<T> new_(A0&&);
template<class T,class A0,class A1> static inline Ref<T> new_(A0&&, A1&&);
template<class T,class A0,class A1,class A2> static inline Ref<T> new_(A0&&, A1&&, A2&&);
template<class T,class A0,class A1,class A2,class A3> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&);
template<class T,class A0,class A1,class A2,class A3,class A4> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&, A4&&);
template<class T,class A0,class A1,class A2,class A3,class A4,class A5> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&, A4&&, A5&&);
template<class T,class A0,class A1,class A2,class A3,class A4,class A5,class A6> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&, A4&&, A5&&, A6&&);
template<class T,class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&, A4&&, A5&&, A6&&, A7&&);
template<class T,class A0,class A1,class A2,class A3,class A4,class A5,class A6,class A7,class A8> static inline Ref<T> new_(A0&&, A1&&, A2&&, A3&&, A4&&, A5&&, A6&&, A7&&, A8&&);

#define GEODE_NEW_FRIEND \
  template<class _T> friend Ref<_T> geode::new_(); \
  template<class _T,class _A0> friend Ref<_T> geode::new_(_A0&&); \
  template<class _T,class _A0,class _A1> friend Ref<_T> geode::new_(_A0&&, _A1&&); \
  template<class _T,class _A0,class _A1,class _A2> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&, _A5&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5,class _A6> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&, _A5&&, _A6&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5,class _A6,class _A7> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&, _A5&&, _A6&&, _A7&&); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5,class _A6,class _A7,class _A8> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&, _A5&&, _A6&&, _A7&&, _A8&&);

#endif

}
