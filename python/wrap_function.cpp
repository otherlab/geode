//#####################################################################
// Function wrap_function
//#####################################################################
#include <other/core/python/wrap_function.h>
namespace other {

struct PythonFunction {
  PyObject_HEAD
  FunctionWrapper wrapper;
  void* wrapped;

  static PyTypeObject pytype; 

  static PyObject* call(PyObject* self, PyObject* args, PyObject* kwds) {
    PythonFunction* self_ = (PythonFunction*)self; 
    if (kwds && PyDict_Size(kwds)) {
      PyErr_SetString(PyExc_TypeError,"function takes no keyword arguments");
      return 0;
    }
    return self_->wrapper(args,self_->wrapped);
  }

  static void dealloc(PyObject* self) {
    self->ob_type->tp_free(self);
  }
};

PyTypeObject PythonFunction::pytype = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                       // ob_size
  "other.Function",        // tp_name
  sizeof(PythonFunction),  // tp_basicsize
  0,                       // tp_itemsize
  PythonFunction::dealloc, // tp_dealloc
  0,                       // tp_print
  0,                       // tp_getattr
  0,                       // tp_setattr
  0,                       // tp_compare
  0,                       // tp_repr
  0,                       // tp_as_number
  0,                       // tp_as_sequence
  0,                       // tp_as_mapping
  0,                       // tp_hash
  PythonFunction::call,    // tp_call
  0,                       // tp_str
  0,                       // tp_getattro
  0,                       // tp_setattro
  0,                       // tp_as_buffer
  Py_TPFLAGS_DEFAULT,      // tp_flags
  "Free function wrapper"  // tp_doc
};

PyObject* wrap_function_helper(const char* name, FunctionWrapper wrapper, void* function) {
  // Allocate a function wrapper
  PyTypeObject* type=&PythonFunction::pytype;
  PythonFunction* f=(PythonFunction*)type->tp_alloc(type,0);
  if (!f) throw std::bad_alloc();

  // fill in fields
  f->wrapper = wrapper;
  f->wrapped = function;

  // all done
  return (PyObject*)f;
}

}
using namespace other;

void wrap_python_function() {
  if (PyType_Ready(&PythonFunction::pytype)<0)
    return;

  // PythonFunction can't be created from python, so no need to add it to the module
}
