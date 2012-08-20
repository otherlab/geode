//#####################################################################
// Class Field
//#####################################################################
//
// An array indexed by a handle type to distinguish between different kinds of fields
//
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
namespace other {

template<class T,class Id> struct HasCheapCopy<Field<T,Id> >:public mpl::true_{};

template<class T,class Id> static inline PyObject* to_python(const Field<T,Id>& field) {
  return to_python(field.flat);
}

template<class T,class Id> struct FromPython<Field<T,Id> >{static inline Field<T,Id> convert(PyObject* object) {
  return Field<T,Id>(from_python<Array<T> >(object));
}};

template<class T,class Id>
class Field {
public:
  typedef typename boost::remove_const<T>::type Element;
  static const bool is_const=boost::is_const<T>::value;
  typedef T& result_type;

  Array<T> flat;

  Field() {}

  Field(const Field<const Element,Id>& source)
    : flat(source.flat) {}

  explicit Field(int n, bool initialize=true)
    : flat(n,initialize) {}

  template<class TA> explicit Field(TA& source)
    : flat(source) {}

  template<class TA> explicit Field(const TA& source)
    : flat(source) {}

  int size() const {
    return flat.size();
  }

  T& operator[](Id i) const {
    return flat[i.idx()];
  }

  bool valid(Id i) const {
    return flat.valid(i.idx());
  }
};

}
