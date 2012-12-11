#pragma once

#include <other/core/vector/Vector.h>
namespace other {

Vector<real,3> hsv_to_rgb(Vector<real,3> cin) OTHER_CORE_EXPORT;
Vector<real,3> rgb_to_hsv(Vector<real,3> const &cin) OTHER_CORE_EXPORT;

static inline Vector<unsigned char,4> to_byte_color(Vector<real,3> const &c, real alpha = 1) {
  return Vector<unsigned char,4>(vec(255*c[0],255*c[1],255*c[2],255*alpha));
}

static inline Vector<real,3> wheel_color(real hue) {
  return hsv_to_rgb(Vector<real,3>(hue,1,1));
}

static inline Vector<real,3> wheel_color(int i, int n) {
  return wheel_color(real(i)/n);
}

}
