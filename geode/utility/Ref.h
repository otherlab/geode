// Ref: A shared pointer that is always nonnull
#pragma once

// Ref always contains an object, and it is impossible to construct an empty Ref.  Use Ptr
// if optional emptiness is desired.  Refs can be directly via new_<T>(...), or from an existing
// object via ref(object).  ref(object) requires enable_shared_from_this, which is always true
// if the class inherits from Object.

#include <geode/utility/Object.h>
#include <geode/utility/new.h>
#include <geode/math/hash.h>
#include <geode/utility/debug.h>
#include <geode/utility/type_traits.h>
#include <iostream>
#include <cassert>
namespace geode {

using std::ostream;

template<class T> class Ref { // T=Object
public:
  GEODE_NEW_FRIEND
  template<class S> friend class Ref;
  template<class S> friend class Ptr;
  typedef T Element;
  struct Safe {};

private:
  shared_ptr<T> p; // Always nonzero
public:

  Ref(const Ref& ref)
    : p(ref.p) {}

  template<class S> Ref(const Ref<S>& ref)
    : p(ref.p) {}

  explicit Ref(const Ptr<T>& ptr)
    : p(ptr.p) {
    GEODE_ASSERT(p);
  }

  // Construct a Ref given an explicit shared_ptr
  template<class S> explicit Ref(const shared_ptr<S>& p)
    : p(p) {
    GEODE_ASSERT(p);
  }

  // Construct a Ref given an explicit shared_ptr, assuming safety
  template<class S> explicit Ref(const shared_ptr<S>& p, Safe)
    : p(p) {}

  // Note: To assign a Ptr to a Ref, use ref(ptr) to convert first.
  Ref& operator=(const Ref& ref) {
    Ref(ref).swap(*this);
    return *this;
  }

  ~Ref() {}

  T& operator*() const {
    return *p.get();
  }

  T* operator->() const {
    return p.get();
  }

  operator T&() const {
    return *p.get();
  }

  T* get() const {
    return p.get();
  }

  const shared_ptr<T>& shared() const {
    return p;
  }

  // Allow conversion to Ref<const T>
  operator Ref<const T>() {
    return Ref<const T>(p);
  }

  void swap(Ref& ref) {
    p.swap(ref.p);
  }

  template<class S> bool operator==(const Ref<S>& o) const { return p==o.p; }
  template<class S> bool operator!=(const Ref<S>& o) const { return p!=o.p; }
  template<class S> bool operator< (const Ref<S>& o) const { return p< o.p; }
  template<class S> bool operator> (const Ref<S>& o) const { return p> o.p; }
  template<class S> bool operator<=(const Ref<S>& o) const { return p<=o.p; }
  template<class S> bool operator>=(const Ref<S>& o) const { return p>=o.p; }

  // Specialize operators to avoid reference counting overhead of converting Ptr<T> to Ref<T>
  template<class S> bool operator==(const Ptr<S>& o) const { return p==o.p; }
  template<class S> bool operator!=(const Ptr<S>& o) const { return p!=o.p; }
  template<class S> bool operator< (const Ptr<S>& o) const { return p< o.p; }
  template<class S> bool operator> (const Ptr<S>& o) const { return p> o.p; }
  template<class S> bool operator<=(const Ptr<S>& o) const { return p<=o.p; }
  template<class S> bool operator>=(const Ptr<S>& o) const { return p>=o.p; }

  Ref<typename remove_const<T>::type> const_cast_() const {
    typedef typename remove_const<T>::type S;
    return Ref<S>(GEODE_SMART_PTR_NAMESPACE::const_pointer_cast<S>(p),Safe());
  }
};

template<class T> static inline typename enable_if<is_base_of<Object,T>,Ref<T>>::type ref(T& object) {
  const auto p = object.shared_from_this();
  assert(p && "object has null shared_from_this.  Was it allocated with new_?");
  return Ref<T>(GEODE_SMART_PTR_NAMESPACE::static_pointer_cast<T>(p),typename Ref<T>::Safe());
}

// Handle const explicitly to be more specific than std::ref
template<class T> static inline typename enable_if<is_base_of<Object,T>,Ref<const T>>::type ref(const T& object) {
  static_assert(is_base_of<Object,T>::value,
                "ref() assumes nonnull enable_shared_from_this, so T must inherit from Object");
  const auto p = object.shared_from_this();
  assert(p && "object has null shared_from_this.  Was it allocated with new_?");
  return Ref<const T>(GEODE_SMART_PTR_NAMESPACE::static_pointer_cast<const T>(p),typename Ref<const T>::Safe());
}

// dynamic_cast for Refs
template<class T,class S> inline Ref<T> dynamic_cast_(const Ref<S>& ref) {
  const auto p = GEODE_SMART_PTR_NAMESPACE::dynamic_pointer_cast<T>(ref.shared());
  if (!p)
    throw std::bad_cast();
  return Ref<T>(p,typename Ref<T>::Safe());
}

// static_cast for Refs
template<class T,class S> inline Ref<T> static_cast_(const Ref<S>& ref) {
  return Ref<T>(GEODE_SMART_PTR_NAMESPACE::static_pointer_cast<T>(ref.shared()),typename Ref<T>::Safe());
}

template<class T> inline ostream& operator<<(ostream& output, const Ref<T>& ref) {
  T& r = ref;
  return output<<"Ref("<<r<<')';
}

template<class T> inline Hash hash_reduce(const Ref<T>& ref) {
  return hash_reduce(ref.get());
}

}

namespace std {
template<class T> void swap(geode::Ref<T>& r1, geode::Ref<T>& r2) {
  r1.swap(r2);
}
}
