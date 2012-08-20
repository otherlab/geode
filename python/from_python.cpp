//#####################################################################
// Function from_python
//#####################################################################
#include <other/core/python/from_python.h>
namespace other {

bool FromPython<bool>::convert(PyObject* object) {
  // Accept only bool/int to avoid confusion
  if (!PyInt_Check(object))
      throw_type_error(object,&PyInt_Type);
  return PyInt_AS_LONG(object)!=0;
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

unsigned long long FromPython<unsigned long long>::convert(PyObject* object) {
  unsigned long long i = PyInt_AsUnsignedLongLongMask(object);
  if (i==(unsigned long long)-1 && PyErr_Occurred()) throw_python_error();
  // Check that the conversion was exact
  PyObject* y = PyLong_FromUnsignedLongLong(i);
  if (!y) throw_python_error();
  int eq = PyObject_RichCompareBool(object,y,Py_EQ);
  Py_DECREF(y);
  if (eq<0) throw_python_error();
  if (!eq) {
    PyErr_Format(PyExc_TypeError,"expected unsigned long long, got %llu",i);
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
  const char* string=PyString_AsString(object);
  if (!string) throw_python_error();
  return string;
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

}
