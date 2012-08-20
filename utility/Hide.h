// A convenience utility for hiding a type from use in overload resolution
#pragma once

namespace other {

template<class T> struct Hide { typedef T type; };

}
