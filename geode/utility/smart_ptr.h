// Pull scoped_ptr and shared_ptr out of std:: if availabe, otherwise boost::
#pragma once

#include <geode/utility/config.h>

// If we're on clang, check for the right header directly.  If we're not,
// any sufficient recently version of gcc should always have the right header.
#if defined(__clang__) ? GEODE_HAS_INCLUDE(<memory>) : defined(__GNU__)
#include <memory>
namespace geode {
template<class T> using scoped_ptr = std::unique_ptr<T>;
using std::shared_ptr;
using std::weak_ptr;
}
#define GEODE_SMART_PTR_NAMESPACE std
#else
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#define GEODE_SMART_PTR_NAMESPACE boost
namespace geode {
using boost::scoped_ptr;
using boost::shared_ptr;
using boost::weak_ptr;
}
#endif

#include <vector>
#include <geode/utility/stl.h>
namespace geode {

using std::vector;
using GEODE_SMART_PTR_NAMESPACE::const_pointer_cast;
using GEODE_SMART_PTR_NAMESPACE::static_pointer_cast;
using GEODE_SMART_PTR_NAMESPACE::dynamic_pointer_cast;

template<class T> struct is_smart_pointer<shared_ptr<T>> : public mpl::true_ {};

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

}
