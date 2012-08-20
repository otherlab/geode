#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>
#include <other/core/utility/stl.h>
namespace other {

using boost::shared_ptr;

template<class T,class... Args> static inline boost::shared_ptr<T> new_sp(Args&&... args) {
  return shared_ptr<T>(new T(args...));
}

template<class T> static inline boost::shared_ptr<typename remove_reference<T>::type> new_sp_copy(T&& t) {
  return shared_ptr<typename remove_reference<T>::type>(new typename remove_reference<T>::type(other_forward<T>(t)));
}

struct null_sp {
  template<typename T> operator shared_ptr<T>() { return shared_ptr<T>(); }
};

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
