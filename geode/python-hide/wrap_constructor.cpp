//#####################################################################
// Function Wrap_Constructor
//#####################################################################
#include <geode/config.h>
#ifdef GEODE_PYTHON
#include <geode/python/wrap_constructor.h>
namespace geode {

void set_argument_count_error(int desired,PyObject* args,PyObject* kwds) {
  Py_ssize_t size = PyTuple_GET_SIZE(args);
  if (size!=desired) {
    if (!desired)
      PyErr_Format(PyExc_TypeError,"constructor takes no arguments (%zd given)",size);
    else
      PyErr_Format(PyExc_TypeError,"constructor takes %d argument%s (%zd given)",desired,(desired>1?"s":""),size);
  } else
    PyErr_SetString(PyExc_TypeError,"constructor takes no keyword arguments");
}

void handle_constructor_error(PyObject* self,const std::exception& error) {
  // Deallocate self without calling the destructor, since the constructor didn't finish.
  // We don't need to call Py_DECREF since we know that no one else owns self yet.
  free(self);

  // report the error
  set_python_exception(error);
}

}
#endif
