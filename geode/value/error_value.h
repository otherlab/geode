//#####################################################################
// Function error_value
//#####################################################################
#pragma once

#include <geode/value/Value.h>
#include <geode/python/ExceptionValue.h>
namespace geode {

template<class T> class ErrorValue : public Value<T> {
public:
  GEODE_NEW_FRIEND
  typedef Value<T> Base;
private:
  ExceptionValue error;

  ErrorValue(const exception& error)
    : error(error) {}

  void update() const {
    error.throw_();
  }

  void dump(int indent) const {
    printf("%*sErrorValue<%s>\n",2*indent,"",typeid(T).name());
  }
};

// Compute a value that always throws the given exception
template<class T> static inline ValueRef<T> error_value(const exception& error) {
  return ValueRef<T>(new_<ErrorValue<T> >(error));
}

}
