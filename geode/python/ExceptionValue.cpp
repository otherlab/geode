#include <geode/python/config.h>
#include <geode/python/ExceptionValue.h>
#include <geode/python/exceptions.h>
#include <geode/utility/tr1.h>
#include <iostream>
namespace geode {

using std::string;
using std::exception;
using std::type_info;
using std::tr1::unordered_map;

class SavedExceptionBase {
public:
  virtual ~SavedExceptionBase() {}

  virtual void throw_() const = 0;
};

namespace {

template<class Error> class SavedException : public SavedExceptionBase {
public:
  string what; 

  SavedException(const char* what)
    : what(what) {}

  void throw_() const {
    throw Error(what);
  }
};

#ifdef GEODE_PYTHON

template<> class SavedException<PythonError> : public SavedExceptionBase {
public:
  PyObject *type, *value, *traceback;

  SavedException(const char* what) {
    PyErr_Fetch(&type,&value,&traceback);
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type,value,traceback);
  }

  ~SavedException() {
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
  }

  void throw_() const {
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type,value,traceback);
    throw_python_error();
  }
};

#endif

typedef SavedExceptionBase* (*Factory)(const char* what);

template<class Error> SavedExceptionBase* factory(const char* what) {
  return new SavedException<Error>(what);
}

typedef unordered_map<const type_info*,Factory> Factories;

// For now, we have a hard coded set of understood exceptions
Factories make_factories() {
  Factories factories;
  #define LEARN(Error) factories[&typeid(Error)] = factory<Error>;
#ifdef GEODE_PYTHON
  LEARN(PythonError)
#endif
  LEARN(RuntimeError)
  LEARN(IOError)
  LEARN(OSError)
  LEARN(LookupError)
  LEARN(IndexError)
  LEARN(KeyError)
  LEARN(TypeError)
  LEARN(ValueError)
  LEARN(NotImplementedError)
  LEARN(AssertionError)
  LEARN(ArithmeticError)
  LEARN(OverflowError)
  LEARN(ZeroDivisionError)
  #undef LEARN
  return factories;
}

Factories factories = make_factories();

}

ExceptionValue::ExceptionValue(const exception& e) {
  const char* what = e.what();
  Factories::const_iterator it = factories.find(&typeid(e));
  if (it != factories.end())
    error.reset(it->second(what));
  else
    error.reset(new SavedException<RuntimeError>(what));
}

void ExceptionValue::throw_() const {
  if (error)
    error->throw_();
}

}
