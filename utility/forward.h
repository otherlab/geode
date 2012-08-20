//#####################################################################
// Header utility/forward
//#####################################################################
#pragma once

namespace other {

// Convenience struct for marking that function semantics have changed
struct Mark {};

// A list of types
template<class... Args> struct Types {
  typedef Types type;
};

}
