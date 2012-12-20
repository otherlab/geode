//#####################################################################
// Header Python/Forward
//#####################################################################
#pragma once

#include <other/core/python/config.h>
#include <other/core/utility/config.h>

namespace other {

class Object;
struct Buffer;
template<class T=PyObject> class Ref;
template<class T=PyObject> class Ptr;

template<class T,class Enable=void> struct FromPython; // from_python<T> isn't defined for types by default

// Should appear at the beginning of all mixed python/C++ classes, after public:
#define OTHER_DECLARE_TYPE(export_spec) \
  OTHER_NEW_FRIEND \
  export_spec static other::PyTypeObject pytype;

#ifdef OTHER_VARIADIC

template<class T,class... Args> static inline Ref<T> new_(Args&&... args);
template<class T,class... Args> static other::PyObject* wrapped_constructor(other::PyTypeObject* type,other::PyObject* args,other::PyObject* kwds);

// Macro to declare new_ as a friend
#define OTHER_NEW_FRIEND \
  template<class _T,class... _Args> friend other::Ref<_T> other::new_(_Args&&... args); \
  template<class _T,class... _Args> friend other::PyObject* ::other::wrapped_constructor(other::PyTypeObject* type,other::PyObject* args,other::PyObject* kwds);

#else

template<class T> static inline Ref<T> new_();
template<class T,class A0> static inline Ref<T> new_(A0&& a0);
template<class T,class A0,class A1> static inline Ref<T> new_(A0&& a0, A1&& a1);
template<class T,class A0,class A1,class A2> static inline Ref<T> new_(A0&& a0, A1&& a1, A2&& a2);
template<class T,class A0,class A1,class A2,class A3> static inline Ref<T> new_(A0&& a0, A1&& a1, A2&& a2, A3&& a3);
template<class T,class A0,class A1,class A2,class A3,class A4> static inline Ref<T> new_(A0&& a0, A1&& a1, A2&& a2, A3&& a3, A4&& a4);
template<class T,class A0,class A1,class A2,class A3,class A4,class A5> static inline Ref<T> new_(A0&& a0, A1&& a1, A2&& a2, A3&& a3, A4&& a4, A5&& a5);

template<class T,class Args> struct WrapConstructor;

#define OTHER_NEW_FRIEND \
  template<class _T> friend Ref<_T> other::new_(); \
  template<class _T,class _A0> friend Ref<_T> other::new_(_A0&& a0); \
  template<class _T,class _A0,class _A1> friend Ref<_T> other::new_(_A0&& a0, _A1&& a1); \
  template<class _T,class _A0,class _A1,class _A2> friend Ref<_T> other::new_(_A0&& a0, _A1&& a1, _A2&& a2); \
  template<class _T,class _A0,class _A1,class _A2,class _A3> friend Ref<_T> other::new_(_A0&& a0, _A1&& a1, _A2&& a2, _A3&& a3); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4> friend Ref<_T> other::new_(_A0&& a0, _A1&& a1, _A2&& a2, _A3&& a3, _A4&& a4); \
  template<class _T,class _A0,class _A1,class _A2,class _A3,class _A4,class _A5> friend Ref<_T> other::new_(_A0&& a0, _A1&& a1, _A2&& a2, _A3&& a3, _A4&& a4, _A5&& a5); \
  template<class _T,class _Args> friend struct other::WrapConstructor;

#endif

// Declare an enum to python.  Must have a corresponding call to OTHER_DEFINE_ENUM from enum.h (in a .cpp).
#define OTHER_DECLARE_ENUM(E,EXPORT) \
  EXPORT PyObject* to_python(E value);

} // namespace other
