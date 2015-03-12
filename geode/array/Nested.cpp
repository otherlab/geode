//#####################################################################
// Class Nested
//#####################################################################
#include <geode/array/Nested.h>
#include <geode/array/convert.h>
#include <geode/python/Class.h>
namespace geode {

Array<int> nested_array_offsets(RawArray<const int> lengths) {
  Array<int> offsets(lengths.size()+1,uninit);
  offsets[0] = 0;
  for (int i=0;i<lengths.size();i++) {
    GEODE_ASSERT(lengths[i]>=0);
    offsets[i+1] = offsets[i]+lengths[i];
  }
  return offsets;
}

#ifdef GEODE_PYTHON

static PyTypeObject* nested_array_type;
static PyObject* offsets_string;
static PyObject* flat_string;
static PyObject* empty_tuple;

static void _set_nested_array(PyObject* nested_array) {
  GEODE_ASSERT(PyObject_IsInstance(nested_array,(PyObject*)&PyType_Type));
  nested_array_type = (PyTypeObject*)nested_array;
}

PyObject* nested_array_to_python_helper(PyObject* offsets, PyObject* flat) {
  PyObject* object = 0;
  if (!nested_array_type) {
    PyErr_SetString(PyExc_RuntimeError,"_set_nested_array must be called before calling to_python");
    goto fail;
  }
  object = nested_array_type->tp_new(nested_array_type,empty_tuple,0);
  if (!object)
    goto fail;
  if (PyObject_GenericSetAttr(object,&*offsets_string,offsets))
    goto fail;
  Py_DECREF(offsets);
  offsets = 0;
  if (PyObject_GenericSetAttr(object,&*flat_string,flat))
    goto fail;
  Py_DECREF(flat);
  flat = 0;
  return object;
fail:
  Py_XDECREF(offsets);
  Py_XDECREF(flat);
  Py_XDECREF(object);
  return 0;
}

bool is_nested_array(PyObject* object) {
  GEODE_ASSERT(nested_array_type);
  return PyObject_IsInstance(object,(PyObject*)nested_array_type)!=0;
}

Vector<Ref<>,2> nested_array_from_python_helper(PyObject* object) {
  return vec(steal_ref_check(PyObject_GetAttr(object,&*offsets_string)),
             steal_ref_check(PyObject_GetAttr(object,&*flat_string)));
}

NESTED_CONVERSIONS(int)
NESTED_CONVERSIONS(float)
NESTED_CONVERSIONS(double)
NESTED_CONVERSIONS(Vector<float,2>)
NESTED_CONVERSIONS(Vector<float,3>)
NESTED_CONVERSIONS(Vector<double,2>)
NESTED_CONVERSIONS(Vector<double,3>)

#endif
}

void wrap_nested_array() {
  using namespace geode;

#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_nested_array)

  offsets_string = PyBytes_FromString("offsets");
  if (!offsets_string) throw_python_error();
  flat_string = PyBytes_FromString("flat");
  if (!flat_string) throw_python_error();
  empty_tuple = PyTuple_New(0);
  if (!empty_tuple) throw_python_error();
#endif
}
