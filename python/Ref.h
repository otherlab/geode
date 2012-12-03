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

#include <other/core/python/forward.h>
#include <other/core/python/exceptions.h>
#include <other/core/python/new.h>
#include <other/core/python/to_python.h>
#include <other/core/math/hash.h>
#include <other/core/utility/debug.h>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <iostream>
namespace other {

namespace mpl = boost::mpl;

OTHER_EXPORT void set_self_owner_mismatch();
OTHER_EXPORT void OTHER_NORETURN(throw_self_owner_mismatch());

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
  OTHER_NEW_FRIEND
  template<class S> friend class Ref;
  template<class S> friend class Ptr;
  template<class S> friend Ref<S> steal_ref(S&);

  T* self; // pointers are always nonzero
  PyObject* owner_;

  Ref() {} // used by new_ and python interface code
public:
  typedef T Element;

  Ref(const Ref& ref)
    : self(ref.self), owner_(ref.owner_) {
    OTHER_INCREF(owner_);
  }

  template<class S> Ref(const Ref<S>& ref)
    : self(RefHelper<T,S>::f(ref.self,ref.owner_)), owner_(ref.owner_) {
    BOOST_MPL_ASSERT((mpl::or_<boost::is_same<T,PyObject>,boost::is_base_of<T,S> >));
    OTHER_INCREF(owner_);
  }

  explicit Ref(const Ptr<T>& ptr)
    : self(ptr.get()), owner_(ptr.borrow_owner()) {
    OTHER_ASSERT(self);
    OTHER_INCREF(owner_);
  }

  // Construct a Ref given explicit self and owner pointers
  Ref(T& self,PyObject* owner)
    : self(&self), owner_(owner) {
    OTHER_INCREF(owner_);
  }

  // note: to assign a ptr to a ref, ref=ptr.ref()
  Ref& operator=(const Ref& ref) {
    OTHER_DECREF(owner_);
    self = ref.self;
    owner_ = ref.owner_;
    OTHER_INCREF(owner_);
    return *this;
  }

  ~Ref() {
    OTHER_DECREF(owner_);
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

  bool operator==(const Ref& o) const {
    return self==o.self;
  }

  bool operator!=(const Ref& o) const {
    return self!=o.self;
  }

  bool operator<(const Ref& o) const {
    return self<o.self;
  }

  bool operator>(const Ref& o) const {
    return self>o.self;
  }

  bool operator<=(const Ref& o) const {
    return self<=o.self;
  }

  bool operator>=(const Ref& o) const {
    return self>=o.self;
  }

  Ref<typename boost::remove_const<T>::type> const_cast_() const {
    typedef typename boost::remove_const<T>::type S;
    return Ref<S>(const_cast<S&>(*self),owner_);
  }

/* waiting for C++11
  template<class... Args> OTHER_ALWAYS_INLINE auto operator()(Args&&... args) const
    -> decltype((*self)(other::forward<Args>(args)...)) {
    return (*self)(other::forward<Args>(args)...);
  }
  */
};

template<class T> inline T* ptr_from_python(PyObject* object) {
  BOOST_MPL_ASSERT((boost::is_base_of<Object,T>));
  return (T*)(object+1);
}

template<> inline PyObject* ptr_from_python<PyObject>(PyObject* object) {
  return object;
}

template<class T> static inline Ref<T>
ref(T& object) {
  BOOST_MPL_ASSERT((mpl::or_<boost::is_same<T,PyObject>,boost::is_base_of<Object,T> >));
  return Ref<T>(object,ptr_to_python(&object));
}

template<class T> static inline Ref<T>
steal_ref(T& object) {
  BOOST_MPL_ASSERT((mpl::or_<boost::is_same<T,PyObject>,boost::is_base_of<Object,T> >));
  Ref<T> ref;
  ref.self = &object;
  ref.owner_ = ptr_to_python(&object);
  return ref;
}

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

// conversion from Ref<T> to python
template<class T> static inline PyObject*
to_python(const Ref<T>& ref) {
  PyObject* owner = ref.borrow_owner();
  if (owner != ptr_to_python(&*ref)) {
    set_self_owner_mismatch();
    return 0;
  }
  Py_INCREF(owner);
  return owner;
}

template<class T> static inline Ref<PyObject>
to_python_ref(const T& x) {
  return steal_ref_check(to_python(x));
}

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
template<class T> void swap(other::Ref<T>& r1, other::Ref<T>& r2) {
  r1.swap(r2);
}
}
