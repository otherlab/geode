//#####################################################################
// Python conversion for enums
//#####################################################################
//
// To use, put OTHER_DEFINE_ENUM(E) in a .cpp, then in a wrap() function do
//
//   OTHER_WRAP_ENUM(E)
//   OTHER_ENUM_VALUE(value0)
//   OTHER_ENUM_VALUE(value1)
//   ...
//
// Note: Including enum.h from a header will result in duplicate symbols.
//
//#####################################################################
#pragma once

#include <other/core/python/Class.h>
#include <other/core/utility/config.h>
#include <other/core/utility/format.h>
#include <other/core/utility/tr1.h>
#include <other/core/python/to_python.h>
#include <boost/mpl/void.hpp>
#include <boost/utility/enable_if.hpp>
namespace other {

using std::make_pair;

#ifdef OTHER_PYTHON

// these are just stubs for the compiler to expect the specializations later.
template<class E> class PyEnum : public Object {};
//template<class E> typename boost::enable_if<boost::is_enum<E>,PyObject*>::type to_python(E value); 
//template<class E> E FromPython<E,typename boost::enable_if<boost::is_enum<E>>::type>::convert(PyObject* object); 

// we have to declare this function
#define OTHER_DECLARE_ENUM(E,EXPORT)\
  EXPORT PyObject* to_python(E value);

// for dllimport/export reaons, this has to be used in the library it is exported from, with its export specifier. So
// we declare the generic template here (which is independent of export specs), and this macro declares template 
// specializations with the proper exports.
#define OTHER_DEFINE_ENUM(E,EXPORT) \
template<> class PyEnum<E> : public Object {\
public:\
  OTHER_DECLARE_TYPE(EXPORT)\
  typedef Object Base;\
  EXPORT static unordered_map<int,Ref<PyEnum>> values;\
  const char* name;\
  E value;\
protected:\
  PyEnum(const char* name, E value)\
    : name(name), value(value) {}\
public:\
  static Ref<PyEnum> new_value(const char* name, E value) {\
    Ref<PyEnum> e = new_<PyEnum>(name,value);\
    values.insert(make_pair((int)value,e));\
    return e;\
  }\
  const char* repr() const {\
    return name;\
  }\
};\
\
unordered_map<int,Ref<PyEnum<E>>> PyEnum<E>::values;\
\
EXPORT PyObject* to_python(E value) {\
  auto it = PyEnum<E>::values.find(value);\
  if (it == PyEnum<E>::values.end())\
    throw KeyError(format("invalid %s value %d",PyEnum<E>::pytype.tp_name,value));\
  return to_python(it->second);\
}\
\
template<> EXPORT E FromPython<E,typename boost::enable_if<boost::is_enum<E>>::type>::convert(PyObject* object) {\
  return from_python<const PyEnum<E>&>(object).value;\
}\
OTHER_DEFINE_TYPE(PyEnum<E>);\
//template EXPORT E FromPython<E>::convert(PyObject*);

#define OTHER_ENUM_2(N,E) \
  {typedef PyEnum<E> Self; \
  Class<Self>(N) \
    .repr() \
    ;}

#define OTHER_ENUM(E) OTHER_ENUM_2(#E,E)

#define OTHER_ENUM_VALUE_2(N,V) \
  other::python::add_object(N,PyEnum<decltype(V)>::new_value(N,V));

#define OTHER_ENUM_VALUE(V) OTHER_ENUM_VALUE_2(#V,V)

#else // non-python stubs

#define OTHER_DEFINE_ENUM(E,EXPORT)
#define OTHER_ENUM_2(N,E)
#define OTHER_ENUM(E)
#define OTHER_ENUM_VALUE_2(N,V)
#define OTHER_ENUM_VALUE(V)

#endif
}
