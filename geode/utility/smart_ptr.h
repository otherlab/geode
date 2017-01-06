// Pull scoped_ptr and shared_ptr out of std::
#pragma once

#include <geode/utility/config.h>
#include <geode/utility/forward.h>
#include <geode/utility/move.h>
#include <geode/utility/mpl.h>
#include <geode/utility/type_traits.h>
#include <memory>
#include <vector>
namespace geode {
template<class T> using scoped_ptr = std::unique_ptr<T>;
using std::shared_ptr;
using std::weak_ptr;

using std::vector;
using std::const_pointer_cast;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;

template<class T> struct is_smart_pointer<shared_ptr<T>> : public mpl::true_ {};

template<class T,class... Args> static inline shared_ptr<T> new_sp(Args&&... args) {
  return shared_ptr<T>(new T(args...));
}

template<class T> static inline shared_ptr<typename remove_reference<T>::type> new_sp_copy(T&& t) {
  return shared_ptr<typename remove_reference<T>::type>(new typename remove_reference<T>::type(geode::forward<T>(t)));
}

template<class T> shared_ptr<const vector<T>> new_copy_with_added(const shared_ptr<const vector<T>>& base, const T& t) {
  shared_ptr<vector<T>> result(new vector<T>(*base));
  result->push_back(t);
  return result;
}

template<class T> shared_ptr<vector<T>> new_copy_with_added(const shared_ptr<vector<T>>& base, const T& t) {
  shared_ptr<vector<T>> result(new vector<T>(*base));
  result->push_back(t);
  return result;
}

template<class T> shared_ptr<const vector<T>> new_copy_with_erased(const shared_ptr<const vector<T>>& base, const T& t) {
  shared_ptr<vector<T>> result(new vector<T>(*base));
  typename vector<T>::iterator iter = find(result->begin(),result->end(),t);
  result->erase(iter);
  return result;
}

} // geode namespace
