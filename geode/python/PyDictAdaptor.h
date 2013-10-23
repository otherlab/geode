#pragma once

#include <geode/python/exceptions.h>
namespace geode {

#ifdef GEODE_PYTHON

class PyDictAdaptor {
  const Ref<> dict;
public:

  PyDictAdaptor()
    : dict(steal_ref_check(PyDict_New())) {}

  template<class K,class V> void set(const K& k, const V& v) {
    Ref<> py_k = to_python_ref(k);
    Ref<> py_v = to_python_ref(v);
    PyDict_SetItem(&*dict, &*py_k, &*py_v);
  }

  template<class K> Ptr<> get(const K& k) {
    return ptr(PyDict_GetItem(&*dict, k));    
  }

  const Ref<>& operator*() {
    return dict;
  }
};

#else // non-python stub

class PyDictAdaptor {
public:
  PyDictAdaptor() {}
  template<class K,class V> void set(const K& k, const V& v) { throw_no_python(); }
  template<class K> Ptr<> get(const K& k) { throw_no_python(); }
  const Ref<>& operator*() { throw_no_python(); }
};

#endif
}
