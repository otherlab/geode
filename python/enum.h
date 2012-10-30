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
#include <boost/mpl/void.hpp>
#include <boost/utility/enable_if.hpp>
#include <tr1/unordered_map>
namespace other{

using std::make_pair;
using std::tr1::unordered_map;

template<class E> class PyEnum : public Object {
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;

  static unordered_map<int,Ref<PyEnum>> values;
  const char* name;
  E value;

protected:
  PyEnum(const char* name, E value)
    : name(name), value(value) {}
public:

  static Ref<PyEnum> new_value(const char* name, E value) {
    Ref<PyEnum> e = new_<PyEnum>(name,value);
    values.insert(make_pair((int)value,e));
    return e;
  }

  const char* repr() const {
    return name;
  }
};

template<class E> unordered_map<int,Ref<PyEnum<E>>> PyEnum<E>::values;

template<class E> typename boost::enable_if<boost::is_enum<E>,PyObject*>::type to_python(E value) {
  auto it = PyEnum<E>::values.find(value);
  if (it == PyEnum<E>::values.end())
    throw KeyError(format("invalid %s value %d",PyEnum<E>::pytype.tp_name,value));
  return to_python(it->second);
}

template<class E> E FromPython<E,typename boost::enable_if<boost::is_enum<E>>::type>::convert(PyObject* object) {
  return from_python<const PyEnum<E>&>(object).value;
}

#define OTHER_DEFINE_ENUM(E) \
  template<> OTHER_DEFINE_TYPE(PyEnum<E>) \
  template E FromPython<E>::convert(PyObject*);

#define OTHER_ENUM_2(N,E) \
  {typedef PyEnum<E> Self; \
  Class<Self>(N) \
    .repr() \
    ;}

#define OTHER_ENUM(E) OTHER_ENUM_2(#E,E)

#define OTHER_ENUM_VALUE_2(N,V) \
  other::python::add_object(N,PyEnum<decltype(V)>::new_value(N,V));

#define OTHER_ENUM_VALUE(V) OTHER_ENUM_VALUE_2(#V,V)

}
