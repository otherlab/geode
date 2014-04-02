// Convenience routines for STL containers
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <geode/math/hash.h>
#include <geode/utility/move.h>
#include <geode/utility/pass.h>
#include <geode/utility/unordered.h>
#include <geode/utility/equals.h>
#include <geode/utility/smart_ptr.h>
#include <geode/utility/type_traits.h>
#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include <iostream>
namespace geode {

using std::vector;

template<class T, class U> inline std::ostream &operator<<(std::ostream &os, std::pair<T,U> const &v);
template<class T,class H> inline std::ostream &operator<<(std::ostream &os, unordered_set<T,H> const &v);
template<class T, class U, class H> inline std::ostream &operator<<(std::ostream &os, unordered_map<T,U,H> const &v);
template<class T> inline std::ostream &operator<<(std::ostream &os, std::vector<T> const &v);
template<class T> inline std::ostream &operator<<(std::ostream &os, std::set<T> const &v);
template<class T> inline std::ostream &operator<<(std::ostream &os, std::list<T> const &v);
template<class T> inline std::ostream &operator<<(std::ostream &os, std::deque<T> const &v);
template<class T, class U> inline std::ostream &operator<<(std::ostream &os, std::map<T,U> const &v);

#ifdef GEODE_VARIADIC

template<typename T, typename... Args> typename std::vector<T> make_vector(Args&&... args) {
  typename std::vector<T> result;
  result.reserve(sizeof...(Args));
  GEODE_PASS(result.push_back(geode::forward<Args>(args)));
  return result;
}
template<typename... Args> struct make_vector_result {
  typedef typename remove_const_reference<typename common_type<Args...>::type>::type type;
};

template<typename... Args> typename std::vector<typename make_vector_result<Args...>::type> make_vector(Args&&... args) {
  typename std::vector<typename make_vector_result<Args...>::type> result;
  result.reserve(sizeof...(Args));
  GEODE_PASS(result.push_back(geode::forward<Args>(args)));
  return result;
}

#else // Unpleasant nonvariadic versions

template<class T> vector<T> make_vector(const T& x0)
{vector<T> v;v.push_back(x0);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1)
{vector<T> v;v.push_back(x0);v.push_back(x1);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);v.push_back(x5);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5,const T& x6)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);v.push_back(x5);v.push_back(x6);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5,const T& x6,const T& x7)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);v.push_back(x5);v.push_back(x6);v.push_back(x7);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5,const T& x6,const T& x7,const T& x8)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);v.push_back(x5);v.push_back(x6);v.push_back(x7);v.push_back(x8);return v;}

template<class T> vector<T> make_vector(const T& x0,const T& x1,const T& x2,const T& x3,const T& x4,const T& x5,const T& x6,const T& x7,const T& x8,const T& x9)
{vector<T> v;v.push_back(x0);v.push_back(x1);v.push_back(x2);v.push_back(x3);v.push_back(x4);v.push_back(x5);v.push_back(x6);v.push_back(x7);v.push_back(x8);v.push_back(x9);return v;}

#endif

// Add a bunch of elements to an stl container
template<class D,class S> inline void extend(D& dst, const S& src) {
  dst.insert(src.begin(), src.end());
}

template<class T,class C> inline void extend(std::vector<T>& dst, const C& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

template<class T,class C> inline void extend(std::deque<T>& dst, const C& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

// check if a STL vector contains an element.
template<class T> inline bool contains(const std::vector<T>& v, const T& x) {
  for (auto y : v) {
    if (Equals<T>::eval(y,x))
      return true;
  }
  return false;
}

// check if a STL map contains an element.
template<class K, class V> inline bool contains(const std::map<K,V>& m,const K& idx){
  return m.count(idx)!=0;
}

// check if a STL set contains an element.
template<class K> inline bool contains(const std::set<K>& s,const K& v){
  return s.count(v)!=0;
}

// Remove an element of a vector in constant time, but destroys vector order.
template<class T> inline void remove_lazy(std::vector<T>& v, size_t i) {
  v[i] = v.back();
  v.pop_back();
}

// Remove the first occurence of an element, destroys vector order.
template<class T> inline void remove_first_lazy(std::vector<T>& v, T const &k) {
  typename std::vector<T>::iterator it = std::find(v.begin(), v.end(), k);

  if (it != v.end()) {
    remove_lazy(v, it-v.begin());
  }
}

// return the max_element of a STL container.
template<class T> typename T::value_type const& max(const T& container) {
  assert(!container.empty());
  return *max_element(container.begin(), container.end());
}


// output operators

template<class T, class U>
inline std::ostream &operator<<(std::ostream &os, std::pair<T,U> const &v) {
  return os << '(' << v.first << ", " << v.second << ')';
}

template<class Iterator>
inline std::ostream &print(std::ostream &os, Iterator const &begin, Iterator const &end, char cbegin = '[', char cend = ']', const char *cdelim = ", ") {
  os << cbegin;

  for (Iterator it = begin; it != end; ++it) {
    if (it != begin)
      os << cdelim;
    os << *it;
  }

  return os << cend;
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, std::deque<T> const &v) {
  return print(os, v.begin(), v.end(), '[', ']');
}

template<class T,class H>
inline std::ostream &operator<<(std::ostream &os, unordered_set<T,H> const &v) {
  return print(os, v.begin(), v.end(), '{', '}');
}

template<class T, class U, class H>
inline std::ostream &operator<<(std::ostream &os, unordered_map<T,U,H> const &v) {
  return print(os, v.begin(), v.end(), '{', '}');
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, std::vector<T> const &v) {
  return print(os, v.begin(), v.end(), '[', ']');
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, std::set<T> const &v) {
  return print(os, v.begin(), v.end(), '{', '}');
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, std::priority_queue<T> const &v) {
  return print(os, v.begin(), v.end(), '[', ']');
}

template<class T>
inline std::ostream &operator<<(std::ostream &os, std::list<T> const &v) {
  return print(os, v.begin(), v.end(), '<', '>');
}

template<class T, class U>
inline std::ostream &operator<<(std::ostream &os, std::map<T,U> const &v) {
  return print(os, v.begin(), v.end(), '{', '}');
}

}
namespace GEODE_SMART_PTR_NAMESPACE {

template<class T> static inline geode::Hash hash_reduce(const shared_ptr<T>& p) {
  using geode::hash_reduce;
  return hash_reduce(p.get());
}

}
namespace std {

template<class T, class U> geode::Hash hash_reduce(const std::pair<T,U>&s) {
  return geode::Hash(s.first, s.second);
}

template<class T> geode::Hash hash_reduce(const std::set<T>& s) {
  using geode::hash_reduce;
  int size = s.size();
  auto it = s.begin();
  if(!size) return geode::Hash();
  if(size==1) return geode::Hash(hash_reduce(*it));
  geode::Hash h;
  if (size&1) {
    auto i0 = it++,
         i1 = it++,
         i2 = it++;
    h = geode::Hash(*i0,*i1,*i2);
  } else {
    auto i0 = it++,
         i1 = it++;
    h = geode::Hash(*i0,*i1);
  }
  while(it!=s.end()) {
    auto i0 = it++,
         i1 = it++;
    h = geode::Hash(h,*i0,*i1);
  }
  return h;
}

}
