// boost::optional 1.51 doesn't work with MSVC 11
#pragma once

#ifndef _WIN32

#include <boost/optional.hpp>

namespace geode {

template<class T> struct Optional: public boost::optional<T> {
  Optional() {}

  Optional(const T& v)
    : boost::optional<T>(v) {}

  Optional(const Optional& o)
    : boost::optional<T>(dynamic_cast<const boost::optional<T>&>(o)) {}
};

}

#else // Windows version

#include <geode/utility/safe_bool.h>
#include <geode/utility/Unique.h>
namespace geode {

template<class T> struct Optional {
private:
  Unique<T> value;
public:

  Optional() {}

  Optional(const T& v)
    : value(new T(v)) {}

  Optional(const Optional& o)
    : value(o?new T(*o):0) {}

  void operator=(const T& v) {
    value.reset(new T(v));
  }

  void operator=(const Optional& o) {
    value.reset(o?new T(*o):0);
  }

  void reset() {
    value.reset();
  }

  T& operator*() {
    assert(value);
    return *value;
  }

  const T& operator*() const {
    assert(value);
    return *value;
  }

  T* operator->() {
    assert(value);
    return value.get();
  }

  const T* operator->() const {
    assert(value);
    return value.get();
  }

  operator SafeBool() const {
    return safe_bool(bool(value));
  }
};

}
#endif
