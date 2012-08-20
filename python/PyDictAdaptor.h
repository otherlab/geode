#pragma once


namespace other {

class PyDictAdaptor {
  const Ref<> dict;
public:

  PyDictAdaptor()
    : dict(steal_ref_check(PyDict_New())) {}

  ~PyDictAdaptor() {}

  template<class K, class V> void put(const K& k, const V& v) {
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

}
