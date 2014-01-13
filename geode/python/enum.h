//#####################################################################
// Python conversion for enums
//#####################################################################
//
// To use,
//
// 1. Put GEODE_DEFINE_ENUM(E) in a .cpp.
// 2. In a wrap() function, do
//
//   GEODE_ENUM(E)
//   GEODE_ENUM_VALUE(value0)
//   GEODE_ENUM_VALUE(value1)
//   ...
//
// Note: Including enum.h from a header will result in duplicate symbols.  Due to complicated
// namespace issues, everything (including the enum) must go inside the geode namespace.
//
//#####################################################################
#pragma once

#include <geode/python/Class.h>
#include <geode/utility/config.h>
#include <geode/utility/format.h>
#include <geode/utility/type_traits.h>
#include <geode/utility/tr1.h>
#include <geode/python/to_python.h>
#include <boost/mpl/void.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
namespace geode {

using std::make_pair;

#ifdef GEODE_PYTHON

// These are just stubs for the compiler to expect the specializations later.
template<class E> class PyEnum : public Object {};

// For dllimport/export reaons, this has to be used in the library it is exported from, with its export specifier. So
// we declare the generic template here (which is independent of export specs), and this macro declares template
// specializations with the proper exports.
#define GEODE_DEFINE_ENUM(E,EXPORT) \
  template<> class PyEnum<E> : public Object { \
    typedef int Unused##E; /* E must be unqualified */ \
  public: \
    GEODE_DECLARE_TYPE(EXPORT BOOST_PP_EMPTY()) /* Last bit dodges a Windows compiler bug */ \
    typedef Object Base; \
    EXPORT static unordered_map<int,Ref<PyEnum>> values; \
    const char* name; \
    E value; \
  protected: \
    PyEnum(const char* name, E value) \
      : name(name), value(value) {} \
  public: \
    static Ref<PyEnum> new_value(const char* name, E value) { \
      Ref<PyEnum> e = new_<PyEnum>(name,value); \
      values.insert(make_pair((int)value,e)); \
      return e; \
    } \
    const char* repr() const { \
      return name; \
    } \
  }; \
  \
  unordered_map<int,Ref<PyEnum<E>>> PyEnum<E>::values; \
  \
  EXPORT PyObject* to_python(E value) { \
    auto it = PyEnum<E>::values.find(value); \
    if (it == PyEnum<E>::values.end()) \
      throw KeyError(format("invalid %s value %d",PyEnum<E>::pytype.tp_name,value)); \
    return to_python(it->second); \
  } \
  \
  template<> EXPORT E FromPython<E,typename enable_if<is_enum<E>>::type>::convert(PyObject* object) { \
    return from_python<const PyEnum<E>&>(object).value; \
  } \
  GEODE_DEFINE_TYPE(PyEnum<E>);

#define GEODE_ENUM_2(N,E) \
  {typedef PyEnum<E> Self; \
  Class<Self>(N) \
    .repr() \
    ;}

#define GEODE_ENUM(E) GEODE_ENUM_2(#E,E)

#define GEODE_ENUM_VALUE_2(N,V) \
  geode::python::add_object(N,PyEnum<decltype(V)>::new_value(N,V));

#define GEODE_ENUM_VALUE(V) GEODE_ENUM_VALUE_2(#V,V)

#else // non-python stubs

#define GEODE_DEFINE_ENUM(E,EXPORT)
#define GEODE_ENUM_2(N,E)
#define GEODE_ENUM(E)
#define GEODE_ENUM_VALUE_2(N,V)
#define GEODE_ENUM_VALUE(V)

#endif
}
