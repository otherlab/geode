//#####################################################################
// Class NestedArray
//#####################################################################
#include <other/core/array/NestedArray.h>
#include <other/core/python/Class.h>
namespace other {

Array<int> nested_array_offsets(RawArray<const int> lengths) {
  Array<int> offsets(lengths.size()+1,false);
  offsets[0] = 0;
  for (int i=0;i<lengths.size();i++) {
    OTHER_ASSERT(lengths[i]>=0);
    offsets[i+1] = offsets[i]+lengths[i];
  }
  return offsets;
}

#ifdef OTHER_PYTHON

static PyTypeObject* nested_array_type;
static PyObject* offsets_string;
static PyObject* flat_string;
static PyObject* empty_tuple;

static void _set_nested_array(PyObject* nested_array) {
  OTHER_ASSERT(PyObject_IsInstance(nested_array,(PyObject*)&PyType_Type));
  nested_array_type = (PyTypeObject*)nested_array;
}

template<class T> PyObject* to_python(const NestedArray<T>& array) {
  OTHER_ASSERT(nested_array_type);
  bool success = true;
  if (PyObject* object = nested_array_type->tp_new(nested_array_type,empty_tuple,0)) {
    if (PyObject* offsets = to_python(array.offsets)) {
      success &= !PyObject_GenericSetAttr(object,&*offsets_string,offsets);
      Py_DECREF(offsets);
      if (PyObject* flat = to_python(array.flat)) {
        success &= !PyObject_GenericSetAttr(object,&*flat_string,flat);
        Py_DECREF(flat);
      }
      if(success) return object;
    }
    Py_DECREF(object);
  }
  return 0;
}

template<class T> NestedArray<T> FromPython<NestedArray<T> >::convert(PyObject* object) {
  OTHER_ASSERT(nested_array_type);
  if (!PyObject_IsInstance(object,(PyObject*)nested_array_type))
    throw_type_error(object,nested_array_type);
  NestedArray<T> self;
  PyObject *offsets=0,*flat=0;
  try {
    offsets = PyObject_GetAttr(object,&*offsets_string);
    if (!offsets) throw_python_error();
    const_cast<Array<const int>&>(self.offsets) = from_python<Array<const int>>(offsets);
    flat = PyObject_GetAttr(object,&*flat_string);
    if (!flat) throw_python_error();
    const_cast<Array<T>&>(self.flat) = from_python<Array<T>>(flat);
  } catch (...) {
    Py_XDECREF(offsets);
    Py_XDECREF(flat);
    throw;
  }
  Py_DECREF(offsets); 
  Py_DECREF(flat); 
  return self;
}

#define INSTANTIATE_HELPER(...) \
  template PyObject* to_python<__VA_ARGS__ >(const NestedArray<__VA_ARGS__ >&); \
  template NestedArray<__VA_ARGS__ > FromPython<NestedArray<__VA_ARGS__ > >::convert(PyObject*);
#define INSTANTIATE(...) \
  INSTANTIATE_HELPER(__VA_ARGS__) \
  INSTANTIATE_HELPER(const __VA_ARGS__)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(Vector<real,2>)

#endif
}

void wrap_nested_array() {
  using namespace other;

#ifdef OTHER_PYTHON
  OTHER_FUNCTION(_set_nested_array)

  offsets_string = PyString_FromString("offsets");
  if (!offsets_string) throw_python_error();
  flat_string = PyString_FromString("flat");
  if (!flat_string) throw_python_error();
  empty_tuple = PyTuple_New(0);
  if (!empty_tuple) throw_python_error();
#endif
}
