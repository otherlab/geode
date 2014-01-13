//#####################################################################
// Class Ref
//#####################################################################
//
// Ref is a smart pointer class for managing python objects.
//
// Ref always contains an object, and it is impossible to construct an empty Ref.  Use Ptr if optional emptiness is desired.
//
// Refs can be directly via new_<T>(...), or from an existing objct via ref(object).
//
// Internally, Ref stores both PyObject* and a T* so that
// 1. Most operations work on incomplete types.
// 2. Values can be easily inspected from a debugger.
// 3. It's possible to store a Ref to a subfield of a python object, or to a multiply-inherited baseclass that isn't itself a python object.
//
//#####################################################################
#pragma once

#include <geode/python/forward.h>
#include <geode/python/exceptions.h>
#include <geode/python/new.h>
#include <geode/python/to_python.h>
#include <geode/math/hash.h>
#include <geode/utility/debug.h>
#include <geode/utility/type_traits.h>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/or.hpp>
#include <iostream>
namespace geode {

namespace mpl = boost::mpl;

GEODE_CORE_EXPORT void set_self_owner_mismatch();
GEODE_CORE_EXPORT void GEODE_NORETURN(throw_self_owner_mismatch());

template<class T> static inline Ref<T> ref(T& object);
template<class T> static inline Ref<T> steal_ref(T& object);

template<class T,class S> struct RefHelper{static T* f(S* self,PyObject* owner) {
  return self;
}};

inline PyObject* ptr_to_python(PyObject* object) {
  return object;
}

inline PyObject* ptr_to_python(const Object* object) {
  return (PyObject*)object-1;
}

template<class S> struct RefHelper<PyObject,S>{static PyObject* f(S* self,PyObject* owner) {
  if (owner != ptr_to_python(self))
    throw_self_owner_mismatch();
  return owner;
}};

template<class T> // T=PyObject
class Ref {
  GEODE_NEW_FRIEND
  template<class S> friend class Ref;
  template<class S> friend class Ptr;
  template<class S> friend Ref<S> steal_ref(S&);
  struct Steal {};

  T* self; // pointers are always nonzero
  PyObject* owner_;

  // Used by new_ and python interface code
  Ref(T& self, PyObject* owner, Steal)
    : self(&self), owner_(owner) {}

public:
  typedef T Element;

  Ref(const Ref& ref)
    : self(ref.self), owner_(ref.owner_) {
    GEODE_INCREF(owner_);
  }

  template<class S> Ref(const Ref<S>& ref)
    : self(RefHelper<T,S>::f(ref.self,ref.owner_)), owner_(ref.owner_) {
    BOOST_MPL_ASSERT((mpl::or_<is_same<T,PyObject>,is_base_of<T,S> >));
    GEODE_INCREF(owner_);
  }

  explicit Ref(const Ptr<T>& ptr)
    : self(ptr.get()), owner_(ptr.borrow_owner()) {
    GEODE_ASSERT(self);
    GEODE_INCREF(owner_);
  }

  // Construct a Ref given explicit self and owner pointers
  Ref(T& self, PyObject* owner)
    : self(&self), owner_(owner) {
    GEODE_INCREF(owner_);
  }

  // note: to assign a ptr to a ref, ref=ptr.ref()
  Ref& operator=(const Ref& ref) {
    Ref(ref).swap(*this);
    return *this;
  }

  ~Ref() {
    GEODE_DECREF(owner_);
  }

  T& operator*() const {
    return *self;
  }

  T* operator->() const {
    return self;
  }

  operator T&() const {
    return *self;
  }

  T* get() const {
    return self;
  }

  // Allow conversion to Ref<const T>
  operator Ref<const T>() {
    return Ref<const T>(*self, owner_);
  }

  PyObject* borrow_owner() const {
    return owner_;
  }

  void swap(Ref& ref) {
    std::swap(self,ref.self);
    std::swap(owner_,ref.owner_);
  }

  template<class S> bool operator==(const Ref<S>& o) const { return self==o.self; }
  template<class S> bool operator!=(const Ref<S>& o) const { return self!=o.self; }
  template<class S> bool operator< (const Ref<S>& o) const { return self< o.self; }
  template<class S> bool operator> (const Ref<S>& o) const { return self> o.self; }
  template<class S> bool operator<=(const Ref<S>& o) const { return self<=o.self; }
  template<class S> bool operator>=(const Ref<S>& o) const { return self>=o.self; }

  // Specialize operators to avoid reference counting overhead of converting Ptr<T> to Ref<T>
  template<class S> bool operator==(const Ptr<S>& o) const { return self==o.self; }
  template<class S> bool operator!=(const Ptr<S>& o) const { return self!=o.self; }
  template<class S> bool operator< (const Ptr<S>& o) const { return self< o.self; }
  template<class S> bool operator> (const Ptr<S>& o) const { return self> o.self; }
  template<class S> bool operator<=(const Ptr<S>& o) const { return self<=o.self; }
  template<class S> bool operator>=(const Ptr<S>& o) const { return self>=o.self; }

  Ref<typename remove_const<T>::type> const_cast_() const {
    typedef typename remove_const<T>::type S;
    return Ref<S>(const_cast<S&>(*self),owner_);
  }

/* Waiting for C++11 to work with PCL
  template<class... Args> GEODE_ALWAYS_INLINE auto operator()(Args&&... args) const
    -> decltype((*self)(geode::forward<Args>(args)...)) {
    return (*self)(geode::forward<Args>(args)...);
  }
  */
};

template<class T> inline T* ptr_from_python(PyObject* object) {
  BOOST_MPL_ASSERT((is_base_of<Object,T>));
  return (T*)(object+1);
}

template<> inline PyObject* ptr_from_python<PyObject>(PyObject* object) {
  return object;
}

template<class T> static inline Ref<T>
ref(T& object) {
  BOOST_MPL_ASSERT((mpl::or_<is_same<T,PyObject>,is_base_of<Object,T> >));
  return Ref<T>(object,ptr_to_python(&object));
}

template<class T> static inline Ref<T>
steal_ref(T& object) {
  BOOST_MPL_ASSERT((mpl::or_<is_same<T,PyObject>,is_base_of<Object,T> >));
  return Ref<T>(object,ptr_to_python(&object),typename Ref<T>::Steal());
}

#ifdef GEODE_PYTHON
template<class T> static inline Ref<T>
ref_check(T* object) {
  if (!object) throw_python_error();
  return ref(*object);
}

template<class T> static inline Ref<T>
steal_ref_check(T* object) {
  if (!object) throw_python_error();
  return steal_ref(*object);
}
#endif

// conversion from Ref<T> to python
template<class T> static inline PyObject*
to_python(const Ref<T>& ref) {
  PyObject* owner = ref.borrow_owner();
  if (owner != ptr_to_python(&*ref)) {
    set_self_owner_mismatch();
    return 0;
  }
  GEODE_INCREF(owner);
  return owner;
}

#ifdef GEODE_PYTHON
template<class T> inline Ref<PyObject>
to_python_ref(const T& x) {
  return steal_ref_check(to_python(x));
}
#endif

// conversion from python to Ref<T>
template<class T> struct FromPython<Ref<T> >{static Ref<T>
convert(PyObject* object) {
  return ref(FromPython<T&>::convert(object));
}};

// conversion from python to Ref<PyObject>
template<> struct FromPython<Ref<PyObject> >{static Ref<PyObject>
convert(PyObject* object) {
  return ref(*object);
}};

// dynamic_cast for Refs
template<class T,class S> inline Ref<T> dynamic_cast_(const Ref<S>& ref) {
  return Ref<T>(dynamic_cast<T&>(*ref),ref.borrow_owner());
}

// static_cast for Refs
template<class T,class S> inline Ref<T> static_cast_(const Ref<S>& ref) {
  return Ref<T>(static_cast<T&>(*ref),ref.borrow_owner());
}

template<class T> inline std::ostream& operator<<(std::ostream& output, const Ref<T>& ref) {
  return output<<"Ref("<<&*ref<<')';
}

template<class T> inline Hash hash_reduce(const Ref<T>& ref) {
  return hash_reduce(&*ref);
}

}

namespace std {
template<class T> void swap(geode::Ref<T>& r1, geode::Ref<T>& r2) {
  r1.swap(r2);
}
}
