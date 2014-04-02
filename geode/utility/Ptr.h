// Ptr: A shared pointer that might be null
#pragma once

// Ptr is the same as Ref, except that it might be null.  Since Ptr can be empty, its dereference
// operators must assert.  Use Ref to avoid this speed penalty if emptiness is not needed.

#include <geode/utility/Ref.h>
namespace geode {

using std::ostream;

template<class T> class Ptr { // T=Object
public:
  GEODE_NEW_FRIEND
  template<class S> friend class Ref;
  template<class S> friend class Ptr;
  typedef T Element;

private:
  shared_ptr<T> p; // May be zero
public:

  Ptr() {}

  Ptr(const Ptr& ptr)
    : p(ptr.p) {}

  template<class S> Ptr(const Ref<S>& ref)
    : p(ref.p) {}

  template<class S> Ptr(const Ptr<S>& ptr)
    : p(ptr.p) {}

  // Construct a Ptr given an explicit shared_ptr
  explicit Ptr(const shared_ptr<T>& p)
    : p(p) {}

  ~Ptr() {}

  Ptr& operator=(const Ptr& ptr) {
    Ptr(ptr).swap(*this);
    return *this;
  }

  template<class S> Ptr& operator=(const Ptr<S>& ptr) {
    Ptr(ptr).swap(*this);
    return *this;
  }

  template<class S> Ptr& operator=(const Ref<S>& ref) {
    Ptr(ref).swap(*this);
    return *this;
  }

  void swap(Ptr& ptr) {
    std::swap(p,ptr.p);
  }

  T& operator*() const {
    GEODE_ASSERT(p);
    return *p.get();
  }

  T* operator->() const {
    GEODE_ASSERT(p);
    return p.get();
  }

  operator T*() const {
    return p.get();
  }

  T* get() const {
    return p.get();
  }

  void clear() {
    p.reset();
  }

  template<class S> bool operator==(const Ptr<S>& o) const { return p==o.p; }
  template<class S> bool operator!=(const Ptr<S>& o) const { return p!=o.p; }
  template<class S> bool operator< (const Ptr<S>& o) const { return p< o.p; }
  template<class S> bool operator> (const Ptr<S>& o) const { return p> o.p; }
  template<class S> bool operator<=(const Ptr<S>& o) const { return p<=o.p; }
  template<class S> bool operator>=(const Ptr<S>& o) const { return p>=o.p; }

  // Specialize operators to avoid reference counting overhead of converting Ref<T> to Ptr<T>
  template<class S> bool operator==(const Ref<S>& o) const { return p==o.p; }
  template<class S> bool operator!=(const Ref<S>& o) const { return p!=o.p; }
  template<class S> bool operator< (const Ref<S>& o) const { return p< o.p; }
  template<class S> bool operator> (const Ref<S>& o) const { return p> o.p; }
  template<class S> bool operator<=(const Ref<S>& o) const { return p<=o.p; }
  template<class S> bool operator>=(const Ref<S>& o) const { return p>=o.p; }

  Ptr<typename remove_const<T>::type> const_cast_() const {
    typedef typename remove_const<T>::type S;
    return Ptr<S>(GEODE_SMART_PTR_NAMESPACE::const_pointer_cast<S>(p));
  }
};

template<class T> static inline Ref<T> ref(Ptr<T>& ptr) {
  return Ref<T>(ptr);
}

template<class T> static inline Ref<T> ref(const Ptr<T>& ptr) {
  return Ref<T>(ptr);
}

template<class T> static inline Ptr<T> ptr(T* object) {
  static_assert(is_base_of<Object,T>::value,
                "ptr() assumes nonnull enable_shared_from_this, so T must inherit from Object");
  if (!object)
    return Ptr<T>();
  const auto p = object->shared_from_this();
  assert(p && "object has null shared_from_this.  Was it allocated with new_?");
  return Ptr<T>(GEODE_SMART_PTR_NAMESPACE::static_pointer_cast<T>(object->shared_from_this()));
}

template<class T> static inline ostream& operator<<(ostream& output, const Ptr<T>& p) {
  return output << p.get();
}

template<class T> static inline Hash hash_reduce(const Ptr<T>& ptr) {
  return hash_reduce(ptr.get());
}

}
