#pragma once
#include <geode/array/Field.h>
#include <geode/structure/Hashtable.h>

namespace geode {

// Provides a bidirectional mapping between a set of elements and a range of Ids.
// A seperate key type is allowed to support searching by a lightweight K (i.g. CircleIntersectionKey) without computing a larger T (i.g. CircleIntersection)
template<class T, class Id, class K=T> struct IdSet {
 protected:
  Hashtable<K, Id> k_to_id;
  Field<T, Id> id_to_t;
 public:
  int size() const { assert(k_to_id.size() == id_to_t.size()); return id_to_t.size(); }
  decltype(id_to_t.id_range()) id_range() const { return id_to_t.id_range(); }

  Id find(const K& k) const { return k_to_id.get_default(k,Id()); }
  
  T const* get_ptr(const K& k) const { 
    Id id = find(k);
    return id.valid() ? &id_to_t[id] : nullptr;
  }

  const T& operator[](const Id id) const { return id_to_t[id]; }

  Id get_or_insert(const K& new_k) {
    Id& new_id = k_to_id.get_or_insert(new_k);
    if(!new_id.valid()) {
      new_id = id_to_t.append(T(new_k));
    }
    return new_id;
  }

  Id insert(const K& new_k, const T& new_t) {
    assert(!k_to_id.contains(new_k));
    const Id new_id = id_to_t.append(new_t);
    k_to_id.insert(new_k, new_id);
    return new_id;
  }

  Id insert(const T& new_t) {
    return insert(K(new_t),new_t);
  }

  void swap_ids(const Id id0, const Id id1) {
    if(id0 == id1)
      return;
    T& t0 = id_to_t[id0];
    T& t1 = id_to_t[id1];
    Id& stored_id0 = k_to_id.get(id0);
    Id& stored_id1 = k_to_id.get(id1);
    swap(t0,t1);
    swap(stored_id0, stored_id1);
  }

  void erase_last(const Id last_id) {
    assert(last_id == id_to_t.id_range().back());
    const T& t = id_to_t[last_id];
    assert(k_to_id.contains(t));
    k_to_id.erase(t);
    id_to_t.flat.pop();
  }

  void lazy_erase(const Id id) {
    const Id last_id = id_to_t.id_range().back();
    swap_ids(id, last_id);
    erase_last(last_id);
  }

  void preallocate(const int max_size) {
    if(max_size > k_to_id.max_size()) {
      k_to_id.resize_table(max_size);
      id_to_t.preallocate(max_size);
    }
  }
};

} // namespace geode