//#####################################################################
// Class Ptr
//#####################################################################
//
// Ptr is a smart pointer class for managing python objects.
//
// Ptr can be empty, so its dereference operators must assert.  Use Ref to avoid this speed penalty if emptiness is not needed.
//
// See Ref for more information.
//
//#####################################################################
#pragma once

#include <geode/python/config.h>
#include <geode/python/new.h>
#include <geode/python/Ref.h>
namespace geode {

using std::ostream;

template<class T> // T=PyObject
class Ptr {
  GEODE_NEW_FRIEND
  template<class S> friend class Ptr;
  template<class S> friend class Ref;

  T* self; // may be null
  PyObject* owner_; // may be null
public:
  typedef T Element;

  Ptr()
    : self(0), owner_(0) {}

  Ptr(const Ptr& ptr)
    : self(ptr.self), owner_(ptr.owner_) {
    GEODE_XINCREF(owner_); // xincref checks for null
  }

  template<class S> Ptr(const Ref<S>& ref)
    : self(RefHelper<T,S>::f(ref.self,ref.owner_)), owner_(ref.owner_) {
    GEODE_INCREF(owner_); // owner_ came from a ref, so no need to check for null
  }

  template<class S> Ptr(const Ptr<S>& ptr) {
    BOOST_MPL_ASSERT((mpl::or_<boost::is_same<T,PyObject>,boost::is_base_of<T,S> >));
    if (ptr.self) {
      self = RefHelper<T,S>::f(ptr.self,ptr.owner_);
      owner_ = ptr.owner_;
      GEODE_INCREF(owner_);
    } else {
      self = 0;
      owner_ = 0;
    }
  }

  // Construct a Ptr given explicit self and owner pointers
  Ptr(T* self,PyObject* owner)
    : self(self), owner_(owner) {
    GEODE_XINCREF(owner_);
  }

  ~Ptr() {
    GEODE_XDECREF(owner_);
  }

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
    std::swap(self,ptr.self);
    std::swap(owner_,ptr.owner_);
  }

  T& operator*() const {
    GEODE_ASSERT(self);
    return *self;
  }

  T* operator->() const {
    GEODE_ASSERT(self);
    return self;
  }

  operator T*() const {
    return self;
  }

  T* get() const {
    return self;
  }

  PyObject* borrow_owner() const {
    return owner_;
  }

  void clear() {
    GEODE_XDECREF(owner_);
    self = 0;
    owner_ = 0;
  }

  template<class S> bool operator==(const Ptr<S>& o) const { return self==o.self; }
  template<class S> bool operator!=(const Ptr<S>& o) const { return self!=o.self; }
  template<class S> bool operator< (const Ptr<S>& o) const { return self< o.self; }
  template<class S> bool operator> (const Ptr<S>& o) const { return self> o.self; }
  template<class S> bool operator<=(const Ptr<S>& o) const { return self<=o.self; }
  template<class S> bool operator>=(const Ptr<S>& o) const { return self>=o.self; }

  // Specialize operators to avoid reference counting overhead of converting Ref<T> to Ptr<T>
  template<class S> bool operator==(const Ref<S>& o) const { return self==o.self; }
  template<class S> bool operator!=(const Ref<S>& o) const { return self!=o.self; }
  template<class S> bool operator< (const Ref<S>& o) const { return self< o.self; }
  template<class S> bool operator> (const Ref<S>& o) const { return self> o.self; }
  template<class S> bool operator<=(const Ref<S>& o) const { return self<=o.self; }
  template<class S> bool operator>=(const Ref<S>& o) const { return self>=o.self; }

  Ptr<typename boost::remove_const<T>::type> const_cast_() const {
    typedef typename boost::remove_const<T>::type S;
    return Ptr<S>(const_cast<S*>(self),owner_);
  }
};

template<class T> static inline Ref<T> ref(Ptr<T>& ptr) {
  return Ref<T>(ptr);
}

template<class T> static inline Ref<T> ref(const Ptr<T>& ptr) {
  return Ref<T>(ptr);
}

template<class T> static inline Ptr<T>
ptr(T* object) {
  BOOST_MPL_ASSERT((boost::is_base_of<Object,T>));
  return object ? Ptr<T>(object,ptr_to_python(object)) : Ptr<T>();
}

#ifdef GEODE_PYTHON

// Convert Py_None to 0 as well
template<> inline Ptr<PyObject>
ptr(PyObject* object) {
  return object && object!=Py_None ? Ptr<>(object,object) : Ptr<>();
}

static inline Ptr<> steal_ptr(PyObject* object) {
  Ptr<> p = ptr(object);
  GEODE_XDECREF(object);
  return p;
}

// Conversion from Ptr<T> to python
template<class T> static inline PyObject*
to_python(const Ptr<T>& ptr) {
  if (ptr) {
    PyObject* owner = ptr.borrow_owner();
    if (owner != ptr_to_python(ptr.get())) {
      set_self_owner_mismatch();
      return 0;
    }
    GEODE_INCREF(owner);
    return owner;
  } else
    Py_RETURN_NONE;
}

// Conversion from python to Ptr<T>
template<class T> struct FromPython<Ptr<T> >{static Ptr<T>
convert(PyObject* object) {
  if (object==Py_None)
    return Ptr<T>();
  else
    return FromPython<Ref<T>>::convert(object);
}};

// Conversion from python to Ptr<PyObject>
template<> struct FromPython<Ptr<PyObject> >{static Ptr<PyObject>
convert(PyObject* object) {
  return ptr(object);
}};

#endif

template<class T> static inline ostream& operator<<(ostream& output, const Ptr<T>& p) {
  return output<<p.get();
}

template<class T> inline Hash hash_reduce(const Ptr<T>& ptr) {
  return hash_reduce(ptr.get());
}

}
