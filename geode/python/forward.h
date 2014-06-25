//#####################################################################
// Header Python/Forward
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/utility/config.h>
#include <geode/utility/type_traits.h>
namespace geode {

class Object;
struct Buffer;
template<class T=PyObject> class Ref;
template<class T=PyObject> class Ptr;
template<class T> struct has_to_python;
template<class T> struct has_from_python;

template<class T> struct is_smart_pointer<Ref<T>> : public mpl::true_ {};
template<class T> struct is_smart_pointer<Ptr<T>> : public mpl::true_ {};

template<class T,class Enable=void> struct FromPython; // from_python<T> isn't defined for types by default

// Should appear at the beginning of all mixed python/C++ classes, after public:
#define GEODE_DECLARE_TYPE(export_spec) \
  GEODE_NEW_FRIEND \
  export_spec static geode::PyTypeObject pytype;

#ifdef GEODE_VARIADIC

template<class T,class... Args> static inline Ref<T> new_(Args&&... args);
template<class T,class... Args> static geode::PyObject* wrapped_constructor(geode::PyTypeObject* type,geode::PyObject* args,geode::PyObject* kwds);

// Macro to declare new_ as a friend
#define GEODE_NEW_FRIEND \
  template<class _T,class... _Args> friend geode::Ref<_T> geode::new_(_Args&&... args); \
  template<class _T,class... _Args> friend geode::PyObject* ::geode::wrapped_constructor(geode::PyTypeObject* type,geode::PyObject* args,geode::PyObject* kwds);

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

template<class T,class Args> struct WrapConstructor;

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
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5,class _A6,class _A7,class _A8> friend Ref<_T> geode::new_(_A0&&, _A1&&, _A2&&, _A3&&, _A4&&, _A5&&, _A6&&, _A7&&, _A8&&); \
  template<class _T,class _Args> friend struct geode::WrapConstructor;

#endif

// Declare an enum to python.  Must have a corresponding call to GEODE_DEFINE_ENUM from enum.h (in a .cpp).
#define GEODE_DECLARE_ENUM(E,EXPORT) \
  EXPORT PyObject* to_python(E value);

} // namespace geode
