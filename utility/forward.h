//#####################################################################
// Header utility/forward
//#####################################################################
#pragma once

namespace boost {
template<class T> class shared_ptr;
}
namespace other {

struct Hasher;
template<class T> class Ptr;

// Convenience struct for marking that function semantics have changed
struct Mark {};

// Convenience utility for hiding a type from use in overload resolution
template<class T> struct Hide {
  typedef T type;
};

// A list of types
template<class... Args> struct Types {
  typedef Types type;
};

// Null pointer convenience class
struct null {
  template<class T> operator Ptr<T>() {
    return Ptr<T>();
  }

  template<typename T> operator boost::shared_ptr<T>() {
    return boost::shared_ptr<T>();
  }
};

}
