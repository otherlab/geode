//#####################################################################
// Class NestedField
//#####################################################################
//
// A wrapper around Nested for accessing with ids
//
//#####################################################################
#pragma once

#include <geode/array/Nested.h>
#include <geode/array/RawField.h>

namespace geode {

template<class T, class Id, bool frozen=true>
class NestedField {
 public:
  Nested<T,frozen> raw;

  NestedField() = default;
  NestedField(Nested<T>&& _raw) : raw(_raw) {}
  NestedField(RawField<const int,Id> lengths) : raw(lengths.flat) {}
  NestedField(RawField<const int,Id> lengths, Uninit) : raw(lengths.flat, uninit) {}
  NestedField(const Field<const int, Id> offsets, const Array<T>& flat) : raw(offsets.flat, flat) {}

  template<class S,class Id2> static NestedField empty_like(const NestedField<S,Id2>& other) {
    return NestedField(Nested<T>::empty_like(other.raw));
  }

  template<class S,bool f> static NestedField empty_like(const Nested<S,f>& other) {
    return NestedField(Nested<T>::empty_like(other));
  }

  int size() const { return raw.size(); }
  int size(const Id i) const { return raw.size(i.idx()); }
  bool empty() const { return raw.empty(); }
  bool valid(const Id i) const { return raw.valid(i.idx()); }
  int total_size() const { return raw.total_size(); }
  Field<int, Id> sizes() const { return Field<int,Id>{raw.sizes()}; }
  T& operator()(const Id i, const int j) const { return raw(i.idx(),j); }
  RawArray<T> operator[](const Id i) const { return raw[i.idx()]; }

  // return index into raw.flat for (*this)[i].front()
  int front_offset(const Id i) const { return raw.offsets[i.idx()]; }
  // return index into raw.flat for (*this)[i].back()
  int back_offset(const Id i) const { return raw.offsets[i.idx()+1]-1; }
  // return range of indices into raw.flat for (*this)[i]
  Range<int> offset_range(const Id i) const { return raw.range(i.idx()); }

  Range<IdIter<Id>> id_range() const { return range(IdIter<Id>(Id(0)),IdIter<Id>(Id(size()))); }
};

} // namespace geode
