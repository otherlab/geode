//#####################################################################
// Class RawField
//#####################################################################
//
// An array indexed by a handle type to distinguish between different kinds of fields
//
//#####################################################################
#pragma once

#include <geode/array/RawArray.h>
namespace geode {

template<class T,class Id> struct HasCheapCopy<RawField<T,Id> >:public mpl::true_{};

template<class T,class Id> static inline PyObject* to_python(const RawField<T,Id>& field) {
  return to_python(field.flat);
}

template<class T,class Id> struct FromPython<RawField<T,Id> >{static inline RawField<T,Id> convert(PyObject* object) {
  return RawField<T,Id>(from_python<Array<T> >(object));
}};

template<class T,class Id>
class RawField {
public:
  typedef typename boost::remove_const<T>::type Element;
  static const bool is_const=boost::is_const<T>::value;
  typedef T& result_type;

  RawArray<T> flat;

  RawField() {}

  RawField(const RawField<Element,Id>& source)
    : flat(source.flat) {}

  RawField(const RawField<const Element,Id>& source)
    : flat(source.flat) {}

  RawField(const Field<Element,Id>& source)
    : flat(source.flat) {}

  RawField(const Field<const Element,Id>& source)
    : flat(source.flat) {}

  explicit RawField(RawArray<T> flat)
    : flat(flat) {}

  int size() const {
    return flat.size();
  }

  T& operator[](Id i) const {
    return flat[i.idx()];
  }

  T& operator()(Id i) const { // Allow use as a function
    return flat[i.idx()];
  }

  bool valid(Id i) const {
    return flat.valid(i.idx());
  }

  // Type safe conversion to go from positions in the field back to an Id
  Id ptr_to_id(const T* x) const {
    Id result = Id(int(x - flat.begin()));
    assert(valid(result));
    return result;
  }

  RawField& operator=(const RawField<Element,Id>& source) {
    flat = source.flat;
    return *this;
  }

  RawField& operator=(const RawField<const Element,Id>& source) {
    flat = source.flat;
    return *this;
  }

  Field<Element,Id> copy() const {
    Field<Element,Id> copy;
    copy.flat.copy(flat);
    return copy;
  }
};

}
