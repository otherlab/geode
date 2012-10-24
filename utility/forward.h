//#####################################################################
// Header utility/forward
//#####################################################################
#pragma once

namespace boost {
template<class T> class shared_ptr;
}
namespace other {

template<class T> class Ptr;

// Convenience struct for marking that function semantics have changed
struct Mark {};

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
