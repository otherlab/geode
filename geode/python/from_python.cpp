//#####################################################################
// Function from_python
//#####################################################################
#include <geode/python/from_python.h>
#include <geode/utility/format.h>
namespace geode {

#ifdef GEODE_PYTHON

bool FromPython<bool>::convert(PyObject* object) {
  // Accept only exact integers to avoid confusion
  return FromPython<long>::convert(object)!=0;
}

long FromPython<long>::convert(PyObject* object) {
  long i = PyInt_AsLong(object);
  if (i==-1 && PyErr_Occurred()) throw_python_error();
  // Check that we had an exact integer
  PyObject* y = PyInt_FromLong(i);
  if (!y) throw_python_error();
  int eq = PyObject_RichCompareBool(object,y,Py_EQ);
  Py_DECREF(y);
  if (eq<0) throw_python_error();
  if (!eq) {
    PyErr_Format(PyExc_TypeError,"expected integer, got %s",object->ob_type->tp_name);
    throw_python_error();
  }
  return i;
}

int FromPython<int>::convert(PyObject* object) {
  long i = FromPython<long>::convert(object);
  if (i!=(int)i) {
    PyErr_SetString(PyExc_OverflowError,"int too large to convert to C int");
    throw_python_error();
  }
  return (int)i;
}

int FromPython<unsigned int>::convert(PyObject* object) {
  long i = FromPython<long>::convert(object);
  if (i!=(unsigned int)i) {
    PyErr_SetString(PyExc_OverflowError,"int too large to convert to C int");
    throw_python_error();
  }
  return (unsigned int)i;
}

long long FromPython<long long>::convert(PyObject* object) {
  long long i = (long long)PyInt_AsUnsignedLongLongMask(object);
  if (i==-1 && PyErr_Occurred())
    throw_python_error();
  // Check that the conversion was exact
  PyObject* y = PyLong_FromLongLong(i);
  if (!y) throw_python_error();
  int eq = PyObject_RichCompareBool(object,y,Py_EQ);
  Py_DECREF(y);
  if (eq<0)
    throw_python_error();
  if (!eq) {
    PyErr_SetString(PyExc_TypeError,format("expected long long, got %s",from_python<const char*>(steal_ref_check(PyObject_Str(object)))).c_str());
    throw_python_error();
  }
  return i;
}

unsigned long long FromPython<unsigned long long>::convert(PyObject* object) {
  unsigned long long i = PyInt_AsUnsignedLongLongMask(object);
  if (i==(unsigned long long)-1 && PyErr_Occurred())
    throw_python_error();
  // Check that the conversion was exact
  PyObject* y = PyLong_FromUnsignedLongLong(i);
  if (!y) throw_python_error();
  int eq = PyObject_RichCompareBool(object,y,Py_EQ);
  Py_DECREF(y);
  if (eq<0)
    throw_python_error();
  if (!eq) {
    PyErr_SetString(PyExc_TypeError,format("expected unsigned long long, got %s",from_python<const char*>(steal_ref_check(PyObject_Str(object)))).c_str());
    throw_python_error();
  }
  return i;
}

unsigned long FromPython<unsigned long>::convert(PyObject* object) {
  unsigned long long i = FromPython<unsigned long long>::convert(object);
  if (i!=(unsigned long)i) {
    PyErr_SetString(PyExc_OverflowError,"int too long to convert to C unsigned long");
    throw_python_error();
  }
  return (unsigned long)i;
}

float FromPython<float>::convert(PyObject* object) {
  double d = PyFloat_AsDouble(object);
  if (d==-1 && PyErr_Occurred()) // -1 either means error or that the value really was -1
    throw_python_error();
  return (float)d; // assume double to float conversion works
}

double FromPython<double>::convert(PyObject* object) {
  double d = PyFloat_AsDouble(object);
  if (d==-1 && PyErr_Occurred()) // -1 either means error or that the value really was -1
    throw_python_error();
  return d;
}

const char* FromPython<const char*>::convert(PyObject* object) {
  const char* string = PyString_AsString(object);
  if (!string) throw_python_error();
  return string;
}

string FromPython<string>::convert(PyObject* object) {
  char* buffer;
  Py_ssize_t size;
  if (PyString_AsStringAndSize(object,&buffer,&size))
    throw_python_error();
  return string(buffer,buffer+size);
}

uint8_t FromPython<uint8_t>::convert(PyObject* object) {
  long i = FromPython<long>::convert(object);
  if (i!=(uint8_t)i) {
    PyErr_SetString(PyExc_OverflowError,"int too large to convert to C uint8_t");
    throw_python_error();
  }
  return (uint8_t)i;
}

char FromPython<char>::convert(PyObject* object) {
  const char* string=PyString_AsString(object);
  if (!string)
    throw_python_error();
  if (string[0] && string[1]) {
    PyErr_Format(PyExc_ValueError,"expected null or single character string for conversion to char, got '%s'",string);
    throw_python_error();
  }
  return string[0];
}

#endif
}
