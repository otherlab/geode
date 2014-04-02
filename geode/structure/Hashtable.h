//#####################################################################
// Class Hashtable
// This hashtable is about twice as fast as the tr1::unordered_map.
// A wrapper class for tr1::unordered_map can be found in STLHashtable.h
// in the STLHashtable branch. It can be used to globally replace
// Hashtable if that is desired.
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/array/Array.h>
#include <geode/math/hash.h>
#include <geode/math/integer_log.h>
#include <geode/utility/Ptr.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/type_traits.h>
#include <vector>
namespace geode {

using std::vector;
using std::ostream;
template<class TK,class T> struct HashtableIter;
template<class TK,class T> class Hashtable;

// Entries

enum HashtableEntryState{EntryFree,EntryActive,EntryDeleted};

template<class TK,class T> struct HashtableEntry {
  typename aligned_storage<sizeof(TK),alignment_of<TK>::value>::type TKbuf;
  typename aligned_storage<sizeof(T),alignment_of<T>::value>::type Tbuf;
  HashtableEntryState state;

  void init(const TK& k, const T& v) {
    new(&TKbuf) TK(k);
    new(&Tbuf) T(v);
    state = EntryActive;
  }

  void destroy() {
    if (state == EntryActive) {
      static_cast<const TK*>(static_cast<const void*>(&TKbuf))->~TK();
      static_cast<const T*>(static_cast<const void*>(&Tbuf))->~T();
    }
    state = EntryDeleted;
  }

  ~HashtableEntry() {
    destroy();
  }

  const TK& key() const { return reinterpret_cast<const TK&>(TKbuf); };
  T& data() const { return reinterpret_cast<T&>(const_cast_(Tbuf)); };
  bool active() const { return state==EntryActive; }

  Tuple<const TK,T>& value() { return reinterpret_cast<Tuple<const TK,T>&>(*this); }
  const Tuple<const TK,T>& value() const { return reinterpret_cast<const Tuple<const TK,T>&>(*this); }
};

template<class TK> struct HashtableEntry<TK,Unit> : public Unit {
  typename aligned_storage<sizeof(TK), alignment_of<TK>::value>::type TKbuf;
  HashtableEntryState state;

  void destroy() {
    if (state == EntryActive) {
      static_cast<const TK*>(static_cast<const void*>(&TKbuf))->~TK();
    }
    state = EntryDeleted;
  }

  ~HashtableEntry() {
    destroy();
  }

  void init(const TK& k, Unit) {
    new(&TKbuf) TK(k);
    state = EntryActive;
  }

  const TK& key() const { return reinterpret_cast<const TK&>(TKbuf); }
  Unit& data() { return *this; }
  bool active() const { return state==EntryActive; }

  const TK& value() const { return key(); }
};

// Tables

template<class TK,class T> // T = Unit
class Hashtable {
private:
  typedef HashtableEntry<TK,T> Entry; // doesn't store data if T is Unit
public:
  typedef TK Key;
  typedef T Element;
  typedef HashtableIter<TK,T> iterator;
  typedef HashtableIter<TK,const T> const_iterator;
  typedef typename remove_const_reference<decltype(declval<Entry>().value())>::type value_type;
private:
  vector<Entry> table_;
  int size_;
  int next_resize_;
public:

  explicit Hashtable(const int estimated_max_size=5) {
    initialize_new_table(estimated_max_size);
  }

  Hashtable(const Tuple<>&) { // Allow conversion from empty tuples
    initialize_new_table(5);
  }

  ~Hashtable() {}

  void clean_memory() {
    initialize_new_table(5); // can't resize to zero since table_.size() must be a power of two
  }

  int size() const {
    return size_;
  }

  int max_size() const {
    return int(table_.size());
  }

  int next_resize() const
    {return next_resize_;
  }

  void initialize_new_table(const int estimated_max_size_) {
    next_resize_ = max(5,estimated_max_size_);
    unsigned estimated_table_entries = (unsigned int)(next_resize_*4/3+1); // choose so load is .75
    int number_of_lists = next_power_of_two(estimated_table_entries);
    table_.resize(number_of_lists); // TODO: only resize if significantly reducing the size
    clear();
  }

  void resize_table(const int estimated_max_size_=0) {
    int estimated_max_size = estimated_max_size_;
    if (!estimated_max_size) estimated_max_size = 3*size_/2;
    vector<Entry> old_table;
    table_.swap(old_table);
    initialize_new_table(estimated_max_size);
    for (int h=0;h<(int)old_table.size();h++) {
      Entry& entry = old_table[h];
      if (entry.active()) insert(entry.key(),entry.data());
    }
  }

  // For iterating over a hashtable in parallel
  RawArray<const Entry> table() const {
    return table_;
  }

private:
  int next_index(const int h) const { // Linear probing
    return (h+1)&(table_.size()-1); // power of two so mod is dropping high order bits
  }

  int hash_index(const TK& v) const {
    return hash(v)&(table_.size()-1); // power of two so mod is dropping high order bits
  }

  bool contains(const TK& v,const int h) const {
    for (int i=h;;i=next_index(i)) {
      if (table_[i].state==EntryFree) return false;
      else if (table_[i].active() && table_[i].key()==v) return true;
    }
  }
public:

  T& insert(const TK& v, const T& value) { // Assumes no entry with v exists
    if (size_>next_resize_) resize_table();
    size_++;
    int h = hash_index(v);
    assert(!contains(v,h));
    for (;table_[h].state!=EntryFree;h=next_index(h));
    Entry& entry = table_[h];
    entry.init(v, value);
    return entry.data();
  }

  void insert(const TK& v) { // Assumes no entry with v exists
    insert(v,unit);
  }

  T& get_or_insert(const TK& v, const T& default_=T()) { // inserts the default if key not found
    if (size_>next_resize_) resize_table();
    int h = hash_index(v);
    for (;table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key()==v)
        return table_[h].data();
    size_++;
    Entry& entry = table_[h];
    entry.init(v, default_);
    return entry.data();
  }

  T& operator[](const TK& v) { // inserts the default if key not found
    return get_or_insert(v);
  }

  T* get_pointer(const TK& v) { // returns Null if key not found
    for (int h=hash_index(v);table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key()==v)
        return &table_[h].data();
    return 0;
  }

  const T* get_pointer(const TK& v) const { // returns 0 if key not found
    return const_cast<Hashtable<TK,T>&>(*this).get_pointer(v);
  }

  T& get(const TK& v) { // fails if key not found
    if (T* data=get_pointer(v))
      return *data;
    throw KeyError("Hashtable::get");
  }

  const T& get(const TK& v) const { // fails if key not found
    return const_cast<Hashtable<TK,T>&>(*this).get(v);
  }

  T get_default(const TK& v, const T& default_=T()) const { // returns default_ if key not found
    if (const T* data=get_pointer(v))
      return *data;
    return default_;
  }

  bool contains(const TK& v) const {
    return contains(v,hash_index(v));
  }

  bool get(const TK& v, T& value) const {
    for (int h=hash_index(v);table_[h].state!=EntryFree;h=next_index(h))
       if (table_[h].active() && table_[h].key==v) {
          value = table_[h].data();
          return true;
        }
    return false;
  }

  bool set(const TK& v, const T& value) { // if v doesn't exist insert value, else sets its value, returns whether it added a new entry
    if (size_>next_resize_) resize_table(); // if over load average, have to grow (must do this before computing hash index)
    int h = hash_index(v);
    for (;table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key()==v) {
        table_[h].data() = value;
        return false;
      }
    size_++;
    Entry& entry = table_[h];
    entry.init(v,value);
    return true;
  }

  bool set(const TK& v) { // insert entry if doesn't already exists, returns whether it added a new entry
    return set(v,unit);
  }

  bool erase(const TK& v) { // Erase an element if it exists, returning true if so
    for (int h=hash_index(v);table_[h].state!=EntryFree;h=next_index(h)) // reduce as still are using entries for deletions
      if (table_[h].active() && table_[h].key()==v) {
        table_[h].destroy();
        size_--;
        next_resize_--;
        return true;
      }
    return false;
  }

  void clear() {
    for (int i=0;i<(int)table_.size();i++)
      table_[i].state = EntryFree;
    size_ = 0;
  }

  void swap(const TK& x, const TK& y) { // Swap values at entries x and y; valid if x or y (or both) are not present; efficient for array values
    bool a = contains(x),
         b = contains(y);
    if (a || b) {
      swap(get_or_insert(x),get_or_insert(y));
      if (!a || !b) erase(a?x:y);
    }
  }

  void swap(Hashtable& other) {
    table_.swap(other.table_);
    std::swap(size_,other.size_);
    std::swap(next_resize_,other.next_resize_);
  }

  iterator begin() {
    return iterator(table_,0);
  }

  const_iterator begin() const {
    return const_iterator(table_,0);

  }

  iterator end() {
    return iterator(table_,(int)table_.size());
  }

  const_iterator end() const {
    return const_iterator(table_,(int)table_.size());
  }
};

// Iteration

template<class TK,class T>
struct HashtableIter {
  typedef typename CopyConst<HashtableEntry<TK,typename remove_const<T>::type>,T>::type Entry;

  const RawArray<Entry> table;
  int index;

  HashtableIter(RawArray<Entry> table, const int index_)
    : table(table), index(index_) {
    while (index<table.size() && !table[index].active())
      index++;
  }

  void operator=(const HashtableIter& other) {
    assert(table.data()==other.table.data());
    index = other.index;
  }

  bool operator==(const HashtableIter& other) const {
    return index==other.index; // Assume same table
  }

  bool operator!=(const HashtableIter& other) const {
    return index!=other.index; // Assume same table
  }

  auto operator*() const
    -> decltype(table[index].value()) {
    auto& entry = table[index];
    assert(entry.active());
    return entry.value();
  }

  auto operator->() const
    -> decltype(&operator*()) {
    return &operator*();
  }

  void operator++() {
    index++;
    while (index<table.size() && !table[index].active())
      index++;
  }
};

template<class K> ostream& operator<<(ostream& output, const Hashtable<K>& h) {
  output << "set([";
  bool first = true;
  for (const auto& v : h) {
    if (first) first = false;
    else output << ',';
    output << v;
  }
  return output << "])";
}

template<class K,class V> ostream& operator<<(ostream& output, const Hashtable<K,V>& h) {
  output << '{';
  bool first = true;
  for (const auto& v : h) {
    if (first) first = false;
    else output << ',';
    output << v.x << ':' << v.y;
  }
  return output << '}';
}

}
namespace std {
template<class TK,class T> void swap(geode::Hashtable<TK,T>& hash1,geode::Hashtable<TK,T>& hash2) {
  hash1.swap(hash2);
}
}
