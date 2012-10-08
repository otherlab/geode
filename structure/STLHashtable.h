#pragma once

#include <other/core/array/forward.h>
#include <other/core/array/Array.h>

#include <other/core/math/hash.h>
#include <other/core/math/integer_log.h>
#include <vector>

#include <tr1/unordered_map>
#include <other/core/utility/Hasher.h>

namespace other {

template<class TK,class T> struct STLHashtableIter;

template<class TK,class T> class STLHashtableEntry {
public:
  TK const &key;
  T &data;  
  STLHashtableEntry(TK const &key, T&data): key(key), data(data) {};

  friend struct STLHashtableIter<TK,T>;
  friend struct STLHashtableIter<TK,const T>;
  
  typedef STLHashtableEntry& value_type;
private:
  value_type value() {return *this;}
};

template<class TK> class STLHashtableEntry<TK,unit> : public unit {
public:
  TK const &key;
  STLHashtableEntry(TK const &key, unit &): key(key) {};

  friend struct STLHashtableIter<TK,unit>;
  friend struct STLHashtableIter<TK,const unit>;
  
  typedef TK const & value_type;
private:
  value_type value() const {return key;}
};

template<class TK> class STLHashtableEntry<TK,const unit> : public unit {
public:
  TK const &key;
  STLHashtableEntry(TK const &key, unit const &): key(key) {};

  friend struct STLHashtableIter<TK,unit>;
  friend struct STLHashtableIter<TK,const unit>;
  
  typedef TK const & value_type;
private:
  value_type value() const {return key;}
};


template<class TK, class T> class STLHashtable;

template<class TK,class T>
struct STLHashtableIter {
  // T will be const if this i a const_iterator
  typedef STLHashtableEntry<TK,T> Entry; 

  typedef typename STLHashtable<TK,typename remove_const<T>::type>::Base Base;
  typedef typename Base::iterator base_iterator;
  typedef typename Base::const_iterator base_const_iterator;
  
  // the base type will be a const_iterator if T is const (as returned by the begin() and end() functions in STLHashtable)
  typedef typename boost::mpl::if_<typename boost::is_const<T>::type, base_const_iterator, base_iterator>::type IterType;

  IterType it;
  
  STLHashtableIter(IterType it): it(it) {}

  bool operator==(const STLHashtableIter& other) const {
    return it==other.it;
  }

  bool operator!=(const STLHashtableIter& other) const {
    return it!=other.it;
  }

  typename Entry::value_type operator*() const {
    return Entry(it->first, it->second).value();
  }

  void operator++() {
    it++;
  }
};


template<class TK,class T>
class STLHashtable : private std::tr1::unordered_map<TK,T,Hasher> {
public:
  typedef std::tr1::unordered_map<TK,T,Hasher> Base;
  typedef typename Base::iterator base_iterator;
  typedef typename Base::const_iterator base_const_iterator;

  typedef TK Key;
  typedef T Element;
  typedef STLHashtableIter<TK,T> iterator;
  typedef STLHashtableIter<TK,const T> const_iterator;
  typedef typename STLHashtableEntry<TK,T>::value_type value_type;

  STLHashtable(const int estimated_max_size=5): Base(estimated_max_size) {
  }
  
  ~STLHashtable() {}

  inline void clean_memory() {
    Base::clear();
  }

  inline int size() const {
    return Base::size();
  }

  inline int max_size() const {
    return Base::bucket_count();
  }

  inline int next_resize() const {
    return Base::max_load_factor() * Base::bucket_count();
  }

  inline void initialize_new_table(const int estimated_max_size_) {
    Base::clear();
    Base::reserve(estimated_max_size_);
  }
  
  inline void resize_table(const int estimated_max_size_=0) {
    Base::rehash(estimated_max_size_);
  }
  
  inline RawArray<const value_type> table() const {
    OTHER_NOT_IMPLEMENTED("cannot get contiguous memory from STL unordered_map");
  }
  
  inline T& insert(const TK& v, const T& value) { // Assumes no entry with v exists
    return Base::operator[](v) = value;
  }

  inline void insert(const TK& v) { // Assumes no entry with v exists
    Base::operator[](v);
  }

  inline T& get_or_insert(const TK& v, const T& default_=T()) { // inserts the default if key not found
    base_iterator it = Base::find(v);    
    if (it == Base::end()) {
      return Base::operator[](v) = default_;
    } else {
      return it->second;
    }  
  }

  inline T& operator[](const TK& v) { // inserts the default if key not found
    return get_or_insert(v);
  }

  inline T* get_pointer(const TK& v) { // returns Null if key not found  
    base_iterator it = Base::find(v);
    if (it == Base::end())
      return 0;
    else
      return &it->second;
  }

  inline const T* get_pointer(const TK& v) const { // returns 0 if key not found
    return const_cast<STLHashtable<TK,T>&>(*this).get_pointer(v);
  }

  inline T& get(const TK& v) { // fails if key not found
    base_iterator it = Base::find(v);
    if (it == Base::end())
      throw KeyError("STLHashtable::get");
    else
      return it->second;
  }

  inline const T& get(const TK& v) const { // fails if key not found
    return const_cast<STLHashtable<TK,T>&>(*this).get(v);
  }

  inline T get_default(const TK& v, const T& default_=T()) const { // returns default_ if key not found
    base_const_iterator it = Base::find(v);
    if (it == Base::end())
      return default_;
    else
      return it->second;
  }

  inline bool contains(const TK& v) const {
    return Base::count(v);
  }

  inline bool get(const TK& v, T& value) const {
    base_const_iterator it = Base::find(v);
    if (it == Base::end()) {
      return false;
    } else {
      value = it->second;
      return true;
    }  
  }

  inline bool set(const TK& v, const T& value) { // if v doesn't exist insert value, else sets its value, returns whether it added a new entry
    base_iterator it = Base::find(v);
    if (it == Base::end()) {
      Base::insert(std::make_pair(v, value));
      return true;
    } else {
      it->second = value;
      return false;
    }    
  }

  inline bool set(const TK& v) { // insert entry if doesn't already exists, returns whether it added a new entry
    return set(v,T());
  }

  inline bool erase(const TK& v) { // Erase an element if it exists, returning true if so
    return Base::erase(v);
  }

  inline void clear() {
    return Base::clear();
  }

  inline void swap(const TK& x, const TK& y) { // Swap values at entries x and y; valid if x or y (or both) are not present; efficient for array values
    bool a = contains(x),
         b = contains(y);
    if (a || b) {
      swap(get_or_insert(x),get_or_insert(y));
      if (!a || !b) erase(a?x:y);
    }
  }

  inline void swap(STLHashtable& other) {
    Base::swap(other);
  }

  inline iterator begin() {
    return iterator(Base::begin());
  }

  inline const_iterator begin() const {
    return const_iterator(Base::begin());
    
  }

  inline iterator end() {
    return iterator(Base::end());
  }

  inline const_iterator end() const {
    return const_iterator(Base::end());
  }
};
 
template<class TK, class K=unit> class Hashtable : public STLHashtable<TK,K> { 
public:
  Hashtable(int k = 5): STLHashtable<TK,K>(k) {} 
};
 
}

