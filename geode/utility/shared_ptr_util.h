#pragma once

#include <geode/utility/stl.h>
#include <vector>
namespace geode {

#ifdef GEODE_VARIADIC

template<class T,class... Args> static inline shared_ptr<T> new_sp(Args&&... args) {
  return shared_ptr<T>(new T(args...));
}

#else // Unpleasant nonvariadic versions

template<class T> static inline shared_ptr<T> new_sp() { return shared_ptr<T>(new T()); }
template<class T,class A0> static inline shared_ptr<T> new_sp(A0&& a0) { return shared_ptr<T>(new T(a0)); }
template<class T,class A0,class A1> static inline shared_ptr<T> new_sp(A0&& a0,A1&& a1) { return shared_ptr<T>(new T(a0,a1)); }
template<class T,class A0,class A1,class A2> static inline shared_ptr<T> new_sp(A0&& a0,A1&& a1,A2&& a2) { return shared_ptr<T>(new T(a0,a1,a2)); }

#endif

template<class T> static inline shared_ptr<typename remove_reference<T>::type> new_sp_copy(T&& t) {
  return shared_ptr<typename remove_reference<T>::type>(new typename remove_reference<T>::type(geode::forward<T>(t)));
}

template<class T> shared_ptr<const std::vector<T> > new_copy_with_added(const shared_ptr<const std::vector<T> >& base, const T& t) {
  shared_ptr<std::vector<T> > result(new std::vector<T>(*base));
  result->push_back(t);
  return result;
}

template<class T> shared_ptr<std::vector<T> > new_copy_with_added(const shared_ptr<std::vector<T> >& base, const T& t) {
  shared_ptr<std::vector<T> > result(new std::vector<T>(*base));
  result->push_back(t);
  return result;
}

template<class T> shared_ptr<const std::vector<T> > new_copy_with_erased(const shared_ptr<const std::vector<T> >& base, const T& t) {
  shared_ptr<std::vector<T> > result(new std::vector<T>(*base));
  typename std::vector<T>::iterator iter = find(result->begin(), result->end(), t);
  result->erase(iter);
  return result;
}

}
