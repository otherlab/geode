#pragma once

namespace geode {

#ifdef GEODE_PYTHON

#include <geode/utility/range.h>
#include <geode/python/Ref.h>
#include <geode/python/Object.h>
#include <geode/python/to_python.h>
#include <geode/python/Class.h>

// This makes an iterable/iterator python class out of a range
template<class Iter, bool sub=has_subtract<Iter>::value> class PyRange;
template<class Iter> class PyRange<Iter,true>: public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  GEODE_NEW_FRIEND

  Iter cur, end;

  PyObject *iternext() {
    PyObject *ret = NULL;
    if (cur != end) {
      ret = to_python(*cur);
      ++cur;
    }
    return ret;
  }

  int __len__() const {
    return end-cur;
  }

  static void make_class_helper(const char *Name) {
    typedef PyRange<Iter> Self; Class<Self>(Name).iter().GEODE_METHOD(__len__);
  }

protected:
  PyRange(Range<Iter> const &range): cur(range.lo), end(range.hi) {}
};


// This makes an iterable/iterator python class out of a range
template<class Iter> class PyRange<Iter,false>: public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  GEODE_NEW_FRIEND

  Iter cur, end;

  PyObject *iternext() {
    PyObject *ret = NULL;
    if (cur != end) {
      ret = to_python(*cur);
      ++cur;
    }
    return ret;
  }

  static void make_class_helper(const char *Name) {
    typedef PyRange<Iter> Self; Class<Self>(Name).iter();
  }

protected:
  PyRange(Range<Iter> const &range): cur(range.lo), end(range.hi) {}
};

template<class Iter>
static inline PyObject* to_python(Range<Iter> const &r) {
  return to_python(new_<PyRange<Iter>>(r));
}

// put this in a wrap_... function to define the python iterator class
#define GEODE_PYTHON_RANGE(Iter,Name)\
  { PyRange<Iter>::make_class_helper(Name); }

// use this to allow writing len(R) for a range R in python. The Iter has to
// support subtraction.
#define GEODE_PYTHON_RANGE_WITH_LEN(Iter,Name)\
  { PyRange<Iter>::make_class_helper(Name); }

#else
// Make this a noop if python support is disabled
#define GEODE_PYTHON_RANGE(Iter,Name)

#endif

}

