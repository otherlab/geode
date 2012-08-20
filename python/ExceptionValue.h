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

#include <other/core/utility/config.h>
#include <other/core/utility/safe_bool.h>
#include <boost/shared_ptr.hpp>
namespace other {

using boost::shared_ptr;
class SavedExceptionBase;

class ExceptionValue {
public:
  shared_ptr<SavedExceptionBase> error;

  ExceptionValue() {}
  ExceptionValue(const std::exception& e) OTHER_EXPORT;

  void throw_() const OTHER_EXPORT;

  operator SafeBool() const { // Allow conversion to bool without allowing conversion to T
    return safe_bool(error);
  }
};

}
