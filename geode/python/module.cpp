//#####################################################################
// Module Python
//#####################################################################
#include <geode/utility/config.h>
#define GEODE_IMPORT_NUMPY
#include <geode/python/module.h>
#include <geode/python/enum.h>
#include <geode/python/numpy.h>
#include <geode/python/stl.h>
#include <geode/python/wrap.h>
namespace geode {

#ifdef GEODE_PYTHON

#define ASSERT_NUMPY_TYPE_CONSISTENT(type)\
  BOOST_STATIC_ASSERT_MSG((int)::type == (int)type, "Numpy's and our definition of " #type " doesn't match. Look at numpy-types.h for the problem.")

ASSERT_NUMPY_TYPE_CONSISTENT(NPY_BOOL);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_BYTE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_UBYTE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_SHORT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_USHORT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_INT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_UINT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_LONG);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_ULONG);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_LONGLONG);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_ULONGLONG);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_FLOAT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_DOUBLE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_LONGDOUBLE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_CFLOAT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_CDOUBLE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_CLONGDOUBLE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_OBJECT);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_STRING);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_UNICODE);
ASSERT_NUMPY_TYPE_CONSISTENT(NPY_VOID);

static std::vector<PyObject*> modules;

static PyObject* module() {
  if (modules.empty())
    throw RuntimeError("No current module");
  return modules.back();
}

static void import_geode() {
#ifdef _WIN32
  // On windows, all code is compiled into a single python module, so there's nothing else to import
  return;
#else
  char* name = PyModule_GetName(module());
  if (!name) throw_python_error();
  if (strcmp(name,"geode_wrap")) {
    PyObject* python_str = PyString_FromString("geode");
    if (!python_str) throw_python_error();
    PyObject* python = PyImport_Import(python_str);
    Py_DECREF(python_str);
    if (!python) throw_python_error();
  }
#endif
}

void module_push(const char* name) {
  auto module = Py_InitModule3(name,0,0);
  if (!module)
    throw_python_error();
  modules.push_back(module);
  import_geode();
}

void module_pop() {
  modules.pop_back();
}

template<class TC> static TC convert_test(const TC& c) {
  return c;
}

string str_repr_test(const string& s) {
  return repr(s);
}

namespace python {

void add_object(const char* name, PyObject* object) {
  if (!object) throw PythonError();
  PyModule_AddObject(module(),name,object);
}

}

#else // non-python stubs

namespace python {
void add_object(const char* name, PyObject* object) {}
}

#endif

enum EnumTest { EnumTestA, EnumTestB };
GEODE_DEFINE_ENUM(EnumTest,GEODE_CORE_EXPORT)

}
using namespace geode;
using namespace geode::python;

void wrap_python() {
#ifdef GEODE_PYTHON
  if(strncmp(PY_VERSION,Py_GetVersion(),3)) {
    PyErr_Format(PyExc_ImportError,"python version mismatch: compiled again %s, linked against %s",PY_VERSION,Py_GetVersion());
    throw_python_error();
  }

  GEODE_WRAP(object)
  GEODE_WRAP(python_function)
  GEODE_WRAP(exceptions)
  GEODE_WRAP(test_class)
  GEODE_WRAP(numpy)

  python::function("list_convert_test",convert_test<vector<int> >);
  python::function("set_convert_test",convert_test<unordered_set<int> >);
  python::function("dict_convert_test",convert_test<unordered_map<int,string> >);
  python::function("enum_convert_test",convert_test<EnumTest>);
  GEODE_FUNCTION(str_repr_test)

  GEODE_ENUM(EnumTest)
  GEODE_ENUM_VALUE(EnumTestA)
  GEODE_ENUM_VALUE(EnumTestB)

  // import numpy
  if (_import_array()<0){
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError,"numpy.core.multiarray failed to import");
    throw_python_error();
  }

  python::add_object("real",(PyObject*)PyArray_DescrFromType(NumpyScalar<real>::value));
#endif
}
