//#####################################################################
// Class Field
//#####################################################################
//
// An array indexed by a handle type to distinguish between different kinds of fields
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/mesh/ids.h>
namespace geode {

template<class T,class Id> struct HasCheapCopy<Field<T,Id> >:public mpl::true_{};

template<class T,class Id> static inline typename enable_if<has_to_python<Array<T>>, PyObject*>::type to_python(const Field<T,Id>& field) {
  return to_python(field.flat);
}

template<class T,class Id> struct FromPython<Field<T,Id> >{static inline Field<T,Id> convert(PyObject* object) {
  return Field<T,Id>(from_python<Array<T>>(object));
}};

template<class T,class Id_>
class Field {
private:
  struct Unusable {};
public:
  using Id = Id_;
  typedef typename remove_const<T>::type Element;
  static const bool is_const = geode::is_const<T>::value;
  typedef T& result_type;

  Array<T> flat;

  Field() {}

  Field(const Field& source)
    : flat(source.flat) {}

  Field(typename mpl::if_c<is_const,const Field<Element,Id>&,Unusable>::type source)
    : flat(source.flat) {}

  explicit Field(const int n)
    : flat(n) {}

  explicit Field(const int n, Uninit)
    : flat(n,uninit) {}

  explicit Field(const Array<T>& source)
    : flat(source) {}

  Field(const Hashtable<Id,T>& source, const int size, const T def = T())
    : flat(size,uninit) {
    flat.fill(def);
    for (const auto& p : source)
      flat[p.x.idx()] = p.y;
  }

  Field& operator=(const Field<Element,Id>& source) {
    flat = source.flat;
    return *this;
  }

  Field& operator=(const Field<const Element,Id>& source) {
    flat = source.flat;
    return *this;
  }

  int size() const {
    return flat.size();
  }

  bool empty() const {
    return flat.empty();
  }

  T& operator[](Id i) const {
    return flat[i.idx()];
  }

  T& operator()(Id i) const { // Allow use as a function
    return flat[i.idx()];
  }

  // Extract values of several ids at the same time as a Vector
  Vector<T,2> vec(const Vector<Id,2> ids) const
  { return {(*this)[ids[0]], (*this)[ids[1]]}; }
  Vector<T,3> vec(const Vector<Id,3> ids) const
  { return {(*this)[ids[0]], (*this)[ids[1]], (*this)[ids[2]]}; }
  Vector<T,4> vec(const Vector<Id,4> ids) const
  { return {(*this)[ids[0]], (*this)[ids[1]], (*this)[ids[2]], (*this)[ids[3]]}; }

  bool valid(Id i) const {
    return flat.valid(i.idx());
  }

  // Type safe conversion to go from positions in the field back to an Id
  Id ptr_to_id(const T* x) const {
    Id result = Id(x - flat.begin());
    assert(valid(result));
    return result;
  }

  Range<IdIter<Id>> id_range() const { return range(IdIter<Id>(Id(0)),IdIter<Id>(Id(size()))); }

  Id append(const T& x) GEODE_ALWAYS_INLINE {
    return Id(flat.append(x));
  }

  Id append_assuming_enough_space(const T& x) GEODE_ALWAYS_INLINE {
    return Id(flat.append_assuming_enough_space(x));
  }

  Id append(Uninit) GEODE_ALWAYS_INLINE {
    return Id(flat.append(uninit));
  }

  // A Field is extended with an array, not another field (A different field would use different ids)
  template<class TArray> void extend(const TArray& append_array) {
    flat.extend(append_array);
  }

  void extend(const int n, Uninit) GEODE_ALWAYS_INLINE {
    flat.extend(n,uninit);
  }

  void preallocate(const int m_new) GEODE_ALWAYS_INLINE {
    flat.preallocate(m_new);
  }

  // Grow storage so that max_id will be be a valid index
  void grow_until_valid(const Id max_id) {
    assert(max_id.valid());
    if(size() <= max_id.idx()) {
      flat.resize(max_id.idx() + 1);
    }
    assert(valid(max_id));
  }

  Field<Element,Id> copy() const {
    Field<Element,Id> copy;
    copy.flat.copy(flat);
    return copy;
  }

  Field<Element,Id>& const_cast_() {
    return *(Field<Element,Id>*)this;
  }

  const Field<Element,Id>& const_cast_() const {
    return *(const Field<Element,Id>*)this;
  }

  Element& front() { return flat.front(); }
  const Element& front() const { return flat.front(); }
  Element& back() { return flat.back(); }
  const Element& back() const { return flat.back(); }
};

}
