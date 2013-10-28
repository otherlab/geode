  //#####################################################################
// Class Class
//#####################################################################
//
// Class template Class exposes convenient syntax for registering the type, fields, and methods of a mixed C++/python
// type with python.  The syntax is analogous to boost::python, but is more restrictive and therefore simpler and faster.
//
// The restrictions required to use this mechanism are as follows:

// 1. The class must be a valid PyObject, so that it can be exposed natively to python.
//    The easiest way to accomplish this is to derive from Object.
//
// 2. The class must have a T::Base typedef giving the immediate base, or mpl::void_ if there isn't one.
//
// 3. Any fields to be exposed to python must be one of
//    a. Ref<T> or Ptr<T> (a shareable reference to another object)
//    b. A type which can be converted by value into a native python object, such int, string, or Vector<T,d>.
//    c. A python-shareable type such as Array<T>.
//
// 4. All methods to be exposed to python must have transparent ownership semantics.  This means their arguments and
//    return values should be either one of the above valid field types or a C++ reference to one of them.
//
// 5. Only one overload of each method can be exposed to python.  If overloading is necessary, an extra python-specific
//    overload can be created that dispatches to the original overloads.  This restriction makes method wrapping far,
//    far simpler and significantly faster, since it avoids the need for crazy two phase type conversion schemes.
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/wrap.h>
#include <geode/python/Object.h>
#include <geode/utility/Enumerate.h>
#ifdef GEODE_PYTHON
#include <geode/python/wrap_constructor.h>
#include <geode/python/wrap_field.h>
#include <geode/python/wrap_method.h>
#include <geode/python/wrap_property.h>
#include <geode/python/wrap_call.h>
#endif
#include <boost/mpl/if.hpp>
#include <boost/mpl/void.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_pointer.hpp>
namespace geode {

namespace mpl = boost::mpl;

#ifdef GEODE_PYTHON

GEODE_CORE_EXPORT int trivial_init(PyObject* self,PyObject* args,PyObject* kwds);
GEODE_CORE_EXPORT PyObject* simple_alloc(PyTypeObject* type,Py_ssize_t nitems);
GEODE_CORE_EXPORT void add_descriptor(PyTypeObject* type,const char* name,PyObject* descr);

#define GEODE_BASE_PYTYPE(...) \
  (boost::is_same<__VA_ARGS__::Base,__VA_ARGS__>::value?0:&__VA_ARGS__::Base::pytype)

// Should appear in the .cpp to define the fields declared by GEODE_DECLARE_TYPE(GEODE_SOMETHING_EXPORT)
#define GEODE_DEFINE_TYPE(...) \
  PyTypeObject __VA_ARGS__::pytype = { \
    PyObject_HEAD_INIT(&PyType_Type) \
    0,                                           /* ob_size */\
    "geode_default_name:" __FILE__ ":" #__VA_ARGS__, /* tp_name */\
    sizeof(geode::PyObject)+sizeof(__VA_ARGS__), /* tp_basicsize */\
    0,                                           /* tp_itemsize */\
    geode::Class<__VA_ARGS__>::dealloc,          /* tp_dealloc */\
    0,                                           /* tp_print */\
    0,                                           /* tp_getattr */\
    0,                                           /* tp_setattr */\
    0,                                           /* tp_compare */\
    0,                                           /* tp_repr */\
    0,                                           /* tp_as_number */\
    0,                                           /* tp_as_sequence */\
    0,                                           /* tp_as_mapping */\
    0,                                           /* tp_hash  */\
    0,                                           /* tp_call */\
    0,                                           /* tp_str */\
    0,                                           /* tp_getattro */\
    0,                                           /* tp_setattro */\
    0,                                           /* tp_as_buffer */\
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */\
    "Wrapped C++ class",                         /* tp_doc */\
    0,                                           /* tp_traverse */\
    0,                                           /* tp_clear */\
    0,                                           /* tp_richcompare */\
    0,                                           /* tp_weaklistoffset */\
    0,                                           /* tp_iter */\
    0,                                           /* tp_iternext */\
    0,                                           /* tp_methods */\
    0,                                           /* tp_members */\
    0,                                           /* tp_getset */\
    GEODE_BASE_PYTYPE(__VA_ARGS__),              /* tp_base */\
    0,                                           /* tp_dict */\
    0,                                           /* tp_descr_get */\
    0,                                           /* tp_descr_set */\
    0,                                           /* tp_dictoffset */\
    geode::trivial_init,                         /* tp_init */\
    geode::simple_alloc,                         /* tp_alloc */\
    0,                                           /* tp_new */\
    free,                                        /* tp_free */\
  };

template<class T> static PyObject* str_wrapper(PyObject* self) {
  try {
    return to_python(str(*GetSelf<T>::get(self)));
  } catch (const exception& error) {
    set_python_exception(error);
    return 0;
  }
}

class ClassBase {
protected:
  PyTypeObject* const type;
  GEODE_CORE_EXPORT ClassBase(const char* name,bool visible,PyTypeObject* type,ptrdiff_t offset);
};

// Class goes in an unnamed namespace since for given T, Class<T> should appear in only one object file
namespace {

template<class T>
class Class : public ClassBase {
public:
  typedef typename boost::remove_pointer<decltype(GetSelf<T>::get((PyObject*)0))>::type Self;

  Class(const char* name, bool visible=true)
    : ClassBase(name,visible,&T::pytype,(char*)(typename T::Base*)(T*)1-(char*)(T*)1)
  {}

#ifdef GEODE_VARIADIC

  template<class... Args> Class&
  init(Types<Args...>) {
    if (type->tp_new)
      throw TypeError("Constructor already specified (can't wrap overloaded constructors directly)");
    type->tp_new = wrapped_constructor<T,Args...>;
    return *this;
  }

#else // Nonvariadic version

  template<class Args> Class& init(Args) {
    if (type->tp_new)
      throw TypeError("Constructor already specified (can't wrap overloaded constructors directly)");
    type->tp_new = WrapConstructor<T,Args>::wrap;
    return *this;
  }

#endif

  template<class Field> Class&
  field(const char* name, Field field) {
    add_descriptor(type,name,wrap_field<T>(name,field));
    return *this;
  }

  template<class Method> Class&
  method(const char* name, Method method) {
#ifndef _WIN32
    add_descriptor(type,name,wrap_method<T,Method>(name,method));
#else
    typedef typename DerivedMethod<Self,Method>::type DM;
    add_descriptor(type,name,wrap_method<T,Self>(name,(DM)method));
#endif
    return *this;
  }

  template<class A> Class&
  property(const char* name, A (T::*get)() const) {
    add_descriptor(type,name,wrap_property(name,get));
    return *this;
  }

  template<class A,class B,class R> Class&
  property(const char* name, A (T::*get)() const, R (T::*set)(B)) {
    add_descriptor(type,name,wrap_property(name,get,set));
    return *this;
  }

  Class& str() {
    type->tp_str = str_wrapper<T>;
    return *this;
  }

  Class& repr() {
    type->tp_repr = repr_wrapper;
    return *this;
  }

  Class& getattr() {
    type->tp_getattro = getattro_wrapper; 
    return *this;
  }

  Class& setattr() {
    type->tp_setattro = setattro_wrapper; 
    return *this;
  }

  static void dealloc(PyObject* self) {
    ((T*)(self+1))->~T(); // call destructor
    self->ob_type->tp_free(self);
  }

  Class& call(ternaryfunc call) {
    type->tp_call = call;
    return *this;
  }

private:
  static PyObject* repr_wrapper(PyObject* self) {
    try {
      return to_python(GetSelf<T>::get(self)->repr());
    } catch (const exception& error) {
      set_python_exception(error);
      return 0;
    }
  }

  static PyObject* getattro_wrapper(PyObject* self, PyObject* name) {
    // Try normal field retrieval first
    if (PyObject* v = PyObject_GenericGetAttr(self,name))
      return v;
    else if (!PyErr_ExceptionMatches(PyExc_AttributeError))
      return 0;
    PyErr_Clear();
    // Fall back to custom code
    try {
      return to_python(GetSelf<T>::get(self)->getattr(from_python<const char*>(name)));
    } catch (const exception& error) {
      set_python_exception(error);
      return 0;
    }
  }

  static int setattro_wrapper(PyObject* self, PyObject* name, PyObject* value) {
    // Try normal field assignment first
    if (PyObject_GenericSetAttr(self,name,value)>=0)
      return 0;
    else if (!PyErr_ExceptionMatches(PyExc_AttributeError))
      return -1;
    PyErr_Clear();
    // Fall back to custom code
    try {
      GetSelf<T>::get(self)->setattr(from_python<const char*>(name),value);
      return 0;
    } catch (const exception& error) {
      set_python_exception(error);
      return -1;
    }
  }
};

}

#else // non-python stubs

namespace {

// Should appear in the .cpp to define the fields declared by GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
#define GEODE_DEFINE_TYPE(...) \
  geode::PyTypeObject __VA_ARGS__::pytype = { \
    typeid(__VA_ARGS__).name(),  /* tp_name */ \
    geode::Class<__VA_ARGS__>::dealloc, /* tp_dealloc */ \
  };

template<class T> class Class {
public:
  Class(const char* name, bool visible=true) {}
  template<class Args> Class& init(Args) { return *this; }
  template<class Field> Class& field(const char* name, Field field) { return *this; }
  template<class Method> Class& method(const char* name, Method method) { return *this; }
  template<class A> Class& property(const char* name, A (T::*get)() const) { return *this; }
  template<class A,class B,class R> Class& property(const char* name, A (T::*get)() const, R (T::*set)(B)) { return *this; }
  Class& str() { return *this; }
  Class& repr() { return *this; }
  Class& call() { return *this; }
  Class& setattr() { return *this; }
  Class& getattr() { return *this; }

  static void dealloc(PyObject* self) {
    ((T*)(self+1))->~T(); // call destructor
    free(self);
  }
};

}
#endif

template<class T,class S> static inline typename boost::add_const<S>::type T::*
const_field(S T::* field) {
  return field;
}

#ifdef GEODE_VARIADIC
#define GEODE_INIT(...) init(Enumerate<__VA_ARGS__>())
#else
#define GEODE_INIT(...) init(Types<__VA_ARGS__>())
#endif

#define GEODE_FIELD_2(name,field_) \
  field(name,&Self::field_)

#define GEODE_FIELD(field_) \
  GEODE_FIELD_2(#field_,field_)

#define GEODE_CONST_FIELD_2(name,field_) \
  field(name,const_field(&Self::field_))

#define GEODE_CONST_FIELD(field_) \
  GEODE_CONST_FIELD_2(#field_,field_)

#define GEODE_METHOD_2(name,method_) \
  method(name,&Self::method_)

#define GEODE_METHOD(method_) \
  GEODE_METHOD_2(#method_,method_)

#define GEODE_OVERLOADED_METHOD_2(type,name,method_) \
  method(name,static_cast<type>(&Self::method_))

#define GEODE_OVERLOADED_METHOD(type,method_) \
  GEODE_OVERLOADED_METHOD_2(type,#method_,method_)

#define GEODE_STR() str()

#define GEODE_REPR() repr()

#ifdef GEODE_PYTHON
#ifdef GEODE_VARIADIC
#define GEODE_CALL(...) \
  call(wrap_call<Self,__VA_ARGS__>())
#else
#define GEODE_CALL(...) \
  call(WrapCall<Self,__VA_ARGS__>::wrap())
#endif
#else
#define GEODE_CALL(...) call()
#endif

#define GEODE_GET(name) \
  property(#name,&Self::name)

#define GEODE_GETSET(name) \
  property(#name,&Self::name,&Self::set_##name)

}
