//#####################################################################
// Class ExceptionValue
//#####################################################################
//
// A class for saving and rethrowing of exceptions.
//
// This is useful for implementing caching in situations where exceptions might be thrown.
// ExceptionValues can be safely copied around.
//
//#####################################################################
#pragma once

#include <geode/utility/smart_ptr.h>
namespace geode {

class SavedExceptionBase;

class ExceptionValue {
public:
  typedef ExceptionValue Self;
  shared_ptr<SavedExceptionBase> error;

  ExceptionValue() {}
  GEODE_CORE_EXPORT ExceptionValue(const std::exception& e);

  GEODE_CORE_EXPORT void throw_() const;

  explicit operator bool() const {
    return bool(error);
  }
};

}
