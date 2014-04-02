// C++ equivalents of Python exceptions
//
// Exception translate is done through exact typeids, so exceptions derived from these must be separately
// registered with the translation mechanism.  Unregistered exceptions deriving from std::exception will
// be converted into RuntimeError in python.  If C++ code throws an object not deriving from std::exception,
// it will propagate into the python runtime, and then...explode.
#pragma once

#include <geode/utility/Object.h>
#include <geode/utility/function.h>
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <string>
namespace geode {

using std::string;
using std::ostream;
using std::type_info;
using std::exception;

#define GEODE_SIMPLE_EXCEPTION(Error,Base_) \
  struct GEODE_CORE_CLASS_EXPORT Error : public Base_ { \
    typedef Base_ Base; \
    GEODE_CORE_EXPORT Error(const string& message); \
    GEODE_CORE_EXPORT virtual ~Error() throw (); \
  };

typedef std::runtime_error RuntimeError;

// This list should stay in sync with the instantiations in the .cpp
GEODE_SIMPLE_EXCEPTION(IOError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(OSError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(LookupError,RuntimeError)
  GEODE_SIMPLE_EXCEPTION(IndexError,LookupError)
  GEODE_SIMPLE_EXCEPTION(KeyError,LookupError)
GEODE_SIMPLE_EXCEPTION(TypeError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ValueError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(NotImplementedError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(AssertionError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(AttributeError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ArithmeticError,RuntimeError)
  GEODE_SIMPLE_EXCEPTION(OverflowError,ArithmeticError)
  GEODE_SIMPLE_EXCEPTION(ZeroDivisionError,ArithmeticError)
GEODE_SIMPLE_EXCEPTION(ReferenceError,RuntimeError)
GEODE_SIMPLE_EXCEPTION(ImportError,RuntimeError)

#undef GEODE_SIMPLE_EXCEPTION

// Save an exception for printing or rethrowing later
class SavedException : public Object {
public:
  // Print the saved exception, possibly with traceback information if available
  virtual void print(ostream& output, const string& where) const = 0;

  // Throw the saved exception
  virtual void throw_() const = 0;
};

// Save an exception.  If the exception is Python-like, it will be cleared.
GEODE_CORE_EXPORT Ref<const SavedException> save(const exception& error);

// Register conversions from exception to SavedException for a given typeid.
GEODE_CORE_EXPORT void register_save(const type_info& type,
                                     const function<Ref<const SavedException>(const exception&)>& save);

}
