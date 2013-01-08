#include <other/core/image/color_utils.h>
#include <other/core/array/NdArray.h>
#include <other/core/python/wrap.h>
namespace other {
  
Vector<real,3> hsv_to_rgb(Vector<real,3> cin) {    
  real s = clamp(cin[1],0.,1.);
  real v = clamp(cin[2],0.,1.);
  
  if (s == 0.0)
    return Vector<real,3>(v,v,v);
  else {      
    real h = fmod(cin[0],1);
    if (h < 0)
      h += 1;
    h *= 6;
    int i = min(5,(int)h);
    
    real f = h - i;
    real p = v*(1.-s);
    real q = v*(1.-(s*f));
    real t = v*(1.-(s*(1.-f)));
    switch (i) {
      case 0:  return Vector<real,3>(v,t,p);
      case 1:  return Vector<real,3>(q,v,p);
      case 2:  return Vector<real,3>(p,v,t);
      case 3:  return Vector<real,3>(p,q,v);
      case 4:  return Vector<real,3>(t,p,v);
      default: return Vector<real,3>(v,p,q);
    }
  }
}

Vector<real,3> rgb_to_hsv(Vector<real,3> const &cin) {
  real min = cin.min();
  real max = cin.max();
  real delta = max - min;

  real const &r = cin[0];
  real const &g = cin[1];
  real const &b = cin[2];
  
  Vector<real,3> cout;
  real &h = cout[0];
  real &s = cout[1];
  real &v = cout[2];
  
  v = max;
  if (max == 0) {
    s = 0;
    h = 0;
  } else {
    s = delta / max;
    if (r == max) {
      h = (g - b) / delta;
    } else if (g == max) {
      h = 2. + (b - r) / delta;
    } else {
      h = 4. + (r - g) / delta;
    }
    h /= 6.;
    if (h < 0)
      h += 1;
  }
  
  return cout;
}
  
static inline NdArray<Vector<real,3>> wheel_color_py(NdArray<const real> hues) {
  NdArray<Vector<real,3>> colors(hues.shape,false);
  for (int i=0;i<hues.flat.size();i++)
    colors.flat[i] = wheel_color(hues.flat[i]);
  return colors;
}

}
using namespace other;

void wrap_color_utils() {
  OTHER_FUNCTION_2(wheel_color,wheel_color_py)
}
