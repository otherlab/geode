//#####################################################################
// Class Tuple
//#####################################################################
#pragma once

#include <other/core/structure/Empty.h>
#include <other/core/structure/Singleton.h>
#include <other/core/structure/Pair.h>
#include <other/core/structure/Triple.h>
#include <other/core/structure/Quad.h>
#include <other/core/structure/Quintuple.h>
#include <other/core/utility/Enumerate.h>
#include <other/core/vector/forward.h>
namespace other {

OTHER_CORE_EXPORT void OTHER_NORETURN(throw_tuple_mismatch_error(int expected, int got));

template<class T> static inline Tuple<T>       as_tuple(const Vector<T,1>& v) { return Tuple<T>      (v.x); }
template<class T> static inline Tuple<T,T>     as_tuple(const Vector<T,2>& v) { return Tuple<T,T>    (v.x,v.y); }
template<class T> static inline Tuple<T,T,T>   as_tuple(const Vector<T,3>& v) { return Tuple<T,T,T>  (v.x,v.y,v.z); }
template<class T> static inline Tuple<T,T,T,T> as_tuple(const Vector<T,4>& v) { return Tuple<T,T,T,T>(v.x,v.y,v.z,v.w); }

#ifdef OTHER_VARIADIC

// Convenience and conversion

template<class... Args> static inline Tuple<Args...> tuple(const Args&... args) {
  return Tuple<Args...>(args...);
}

#ifdef OTHER_PYTHON

template<class Tup,class... Enum> static inline PyObject* tuple_to_python_helper(const Tup& src, Types<Enum...>);
template<class Tup,class... Enum> static inline Tup tuple_from_python_helper(PyObject* object, Types<Enum...>);

template<class... Args> PyObject* to_python(const Tuple<Args...>& src) {
  return tuple_to_python_helper(src,Enumerate<Args...>());
}

template<class... Args> struct FromPython<Tuple<Args...>>{static Tuple<Args...> convert(PyObject* object) {
  return tuple_from_python_helper<Tuple<Args...>>(object,Enumerate<Args...>());
}};

#endif

// Tuples of unusual size

template<class T> struct make_reference_const;
template<class T> struct make_reference_const<T&> { typedef const T& type; };

template<class T0,class T1,class T2,class T3,class T4,class... Rest> class Tuple<T0,T1,T2,T3,T4,Rest...> {
  BOOST_STATIC_ASSERT(sizeof...(Rest)>0);
public:
  Tuple<T0,T1,T2,T3,T4> left;
  Tuple<Rest...> right;

  Tuple() {}

  Tuple(const T0& x0, const T1& x1, const T2& x2, const T3& x3, const T4& x4, const Rest&... rest)
    : left(x0,x1,x2,x3,x4), right(rest...) {}

  bool operator==(const Tuple& t) const {
    return left==t.left && right==t.right;
  }

  bool operator!=(const Tuple& t) const {
    return !(*this==t);
  }

  template<int i> auto get()
    -> decltype(choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>()) {
    return choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>();
  }

  template<int i> auto get() const
    -> typename make_reference_const<decltype(choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>())>::type {
    return choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>();
  }
};

// Helper functions for conversion

// Given an ITP<n,T> pair, convert the given element of the tuple to python
template<class A,class Tup> static inline PyObject* convert_tuple_entry(const Tup& src) {
  return to_python(src.template get<A::index>());
}

// Given an ITP<n,T> pair, extract the given index and convert to the desired type
template<class A> static inline auto convert_sequence_item(PyObject* args)
  -> decltype(from_python<typename A::type>((PyObject*)0)) {
  return from_python<typename A::type>(PySequence_Fast_GET_ITEM(args,A::index));
}

#ifdef OTHER_PYTHON

template<class Tup,class... Enum> static inline PyObject* tuple_to_python_helper(const Tup& src, Types<Enum...>) {
  const int n = sizeof...(Enum);
  Vector<PyObject*,n> items(convert_tuple_entry<Enum>(src)...);
  for (auto o : items)
    if (!o)
      goto fail;
  if (auto dst = PyTuple_New(n)) {
    for (int i=0;i<n;i++)
      PyTuple_SET_ITEM(dst,i,items[i]);
    return dst;
  }
fail:
  for (auto o : items)
    Py_XDECREF(o);
  return 0;
}

template<class Tup,class... Enum> static inline Tup tuple_from_python_helper(PyObject* object, Types<Enum...>) {
  const size_t n = sizeof...(Enum);
  Ref<PyObject> seq = steal_ref_check(PySequence_Fast(object,"expected tuple"));
  size_t len = PySequence_Length(&*seq);
  if (len!=n) throw_tuple_mismatch_error(n,len);
  return Tup(convert_sequence_item<Enum>(&*seq)...);
}

#endif

#else // Unpleasant nonvariadic versions

static inline Tuple<> tuple() { return Tuple<>(); }
template<class A0> static inline Tuple<A0> tuple(const A0& a0) { return Tuple<A0>(a0); }
template<class A0,class A1> static inline Tuple<A0,A1> tuple(const A0& a0,const A1& a1) { return Tuple<A0,A1>(a0,a1); }
template<class A0,class A1,class A2> static inline Tuple<A0,A1,A2> tuple(const A0& a0,const A1& a1,const A2& a2) { return Tuple<A0,A1,A2>(a0,a1,a2); }
template<class A0,class A1,class A2,class A3> static inline Tuple<A0,A1,A2,A3> tuple(const A0& a0,const A1& a1,const A2& a2,const A3& a3) { return Tuple<A0,A1,A2,A3>(a0,a1,a2,a3); }
template<class A0,class A1,class A2,class A3,class A4> static inline Tuple<A0,A1,A2,A3,A4> tuple(const A0& a0,const A1& a1,const A2& a2,const A3& a3,const A4& a4) { return Tuple<A0,A1,A2,A3,A4>(a0,a1,a2,a3,a4); }

#ifdef OTHER_PYTHON

#define OTHER_TUPLE_CONVERT(n,ARGS,Args,fromargs,toargs) \
  template<OTHER_REMOVE_PARENS(ARGS)> struct FromPython<Tuple<OTHER_REMOVE_PARENS(Args)>>{static Tuple<OTHER_REMOVE_PARENS(Args)> convert(PyObject* object) { \
    Ref<PyObject> seq = steal_ref_check(PySequence_Fast(object,"expected tuple")); \
    size_t len = PySequence_Length(&*seq); \
    if (len!=n) throw_tuple_mismatch_error(n,len); \
    return Tuple<OTHER_REMOVE_PARENS(Args)> fromargs; \
  }}; \
  \
  template<OTHER_REMOVE_PARENS(ARGS)> PyObject* to_python(const Tuple<OTHER_REMOVE_PARENS(Args)>& src) { \
    Vector<PyObject*,n> items toargs; \
    for (auto o : items) \
      if (!o) \
        goto fail; \
    if (auto dst = PyTuple_New(n)) { \
      for (int i=0;i<n;i++) \
        PyTuple_SET_ITEM(dst,i,items[i]); \
      return dst; \
    } \
  fail: \
    for (auto o : items) { \
      Py_XDECREF(o); \
    } \
    return 0; \
  }

#define TF(i) from_python<A##i>(PySequence_Fast_GET_ITEM(&*seq,i))
#define TT(i) to_python(src.template get<i>())
OTHER_TUPLE_CONVERT(1,(class A0),(A0),(TF(0)),(TT(0)))
OTHER_TUPLE_CONVERT(2,(class A0,class A1),(A0,A1),(TF(0),TF(1)),(TT(0),TT(1)))
OTHER_TUPLE_CONVERT(3,(class A0,class A1,class A2),(A0,A1,A2),(TF(0),TF(1),TF(2)),(TT(0),TT(1),TT(2)))
OTHER_TUPLE_CONVERT(4,(class A0,class A1,class A2,class A3),(A0,A1,A2,A3),(TF(0),TF(1),TF(2),TF(3)),(TT(0),TT(1),TT(2),TT(3)))
OTHER_TUPLE_CONVERT(5,(class A0,class A1,class A2,class A3,class A4),(A0,A1,A2,A3,A4),(TF(0),TF(1),TF(2),TF(3),TF(4)),(TT(0),TT(1),TT(2),TT(3),TT(4)))
#undef OTHER_TUPLE_CONVERT
#undef TT
#undef TF

#endif
#endif

}
