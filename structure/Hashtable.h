//#####################################################################
// Class Hashtable
// This hashtable is about twice as fast as the tr1::unordered_map. 
// A wrapper class for tr1::unordered_map can be found in STLHashtable.h
// in the STLHashtable branch. It can be used to globally replace 
// Hashtable if that is desired.
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/array/Array.h>

#include <other/core/math/hash.h>
#include <other/core/math/integer_log.h>
#include <vector>

namespace other {

using std::vector;
template<class TK,class T> struct HashtableIter;
template<class TK,class T> class Hashtable;

// Entries

enum HashtableEntryState{EntryFree,EntryActive,EntryDeleted};

template<class TK,class T> class HashtableEntry {
public:
  TK key;
  mutable T data;
  
  friend struct HashtableIter<TK,T>;
  friend struct HashtableIter<TK,const T>;
  friend class Hashtable<TK,T>;

private:
  HashtableEntryState state;
  bool active() const {return state==EntryActive;}
  T& data_() const {return data;}
  const HashtableEntry& value() const {return *this;}
};

template<class TK> class HashtableEntry<TK,unit> : public unit {
public:
  TK key;

  friend struct HashtableIter<TK,unit>;
  friend struct HashtableIter<TK,const unit>;
  friend class Hashtable<TK,unit>;

private:
  HashtableEntryState state;
  bool active() const {return state==EntryActive;}
  unit& data_() {return *this;}
  const TK& value() const {return key;}
};

// Tables

template<class TK,class T> // T = unit
class Hashtable {
private:
  typedef HashtableEntry<TK,T> Entry; // doesn't store data if T is unit
public:
  typedef TK Key;
  typedef T Element;
  typedef HashtableIter<TK,T> iterator;
  typedef HashtableIter<TK,const T> const_iterator;
  typedef typename remove_const_reference<decltype(boost::declval<Entry>().value())>::type value_type;
private:
  vector<Entry> table_;
  int size_;
  int next_resize_;
public:

  Hashtable(const int estimated_max_size=5) {
    initialize_new_table(estimated_max_size);
  }

  ~Hashtable() {}

  void clean_memory() {
    initialize_new_table(5); // can't resize to zero since table_.size() must be a power of two
  }

  int size() const {
    return size_;
  }

  int max_size() const {
    return table_.size();
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
      if (entry.active()) insert(entry.key,entry.data_());
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
      else if (table_[i].active() && table_[i].key==v) return true;
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
    entry.key = v;
    entry.state = EntryActive;
    return entry.data_() = value;
  }

  void insert(const TK& v) { // Assumes no entry with v exists
    insert(v,unit());
  }

  T& get_or_insert(const TK& v, const T& default_=T()) { // inserts the default if key not found
    if (size_>next_resize_) resize_table();
    int h = hash_index(v);
    for (;table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key==v) return table_[h].data_();
    size_++;
    Entry& entry = table_[h];
    entry.key = v;
    entry.state = EntryActive;
    entry.data_() = default_;
    return entry.data_();
  }

  T& operator[](const TK& v) { // inserts the default if key not found
    return get_or_insert(v);
  }

  T* get_pointer(const TK& v) { // returns Null if key not found
    for (int h=hash_index(v);table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key==v) return &table_[h].data_();
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
          value = table_[h].data_();
          return true;
        }
    return false;
  }

  bool set(const TK& v, const T& value) { // if v doesn't exist insert value, else sets its value, returns whether it added a new entry
    if (size_>next_resize_) resize_table(); // if over load average, have to grow (must do this before computing hash index)
    int h = hash_index(v);
    for (;table_[h].state!=EntryFree;h=next_index(h))
      if (table_[h].active() && table_[h].key==v) {
        table_[h].data_() = value;
        return false;
      }
    size_++;
    Entry& entry = table_[h];
    entry.key = v;
    entry.state = EntryActive;
    entry.data_() = value;
    return true;
  }

  bool set(const TK& v) { // insert entry if doesn't already exists, returns whether it added a new entry
    return set(v,unit());
  }

  bool erase(const TK& v) { // Erase an element if it exists, returning true if so
    for (int h=hash_index(v);table_[h].state!=EntryFree;h=next_index(h)) // reduce as still are using entries for deletions
      if (table_[h].active() && table_[h].key==v) {
        table_[h].state = EntryDeleted;
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
  typedef HashtableEntry<TK,typename boost::remove_const<T>::type> Entry;

  const RawArray<const Entry> table;
  int index;

  HashtableIter(RawArray<const Entry> table, int index_)
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
    assert(index<table.size() && table[index].active());
    return table[index].value();
  }

  auto operator->() const
    -> decltype(&table[index].value()) {
    assert(index<table.size() && table[index].active());
    return &table[index].value();
  }

  void operator++() {
    index++;
    while (index<table.size() && !table[index].active())
      index++;
  }
};

}
namespace std {
template<class TK,class T> void swap(other::Hashtable<TK,T>& hash1,other::Hashtable<TK,T>& hash2) {
  hash1.swap(hash2);
}
}
