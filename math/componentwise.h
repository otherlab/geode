//#####################################################################
// Header componentwise
//#####################################################################
#pragma once

#include <other/core/math/min.h>
#include <other/core/math/max.h>
namespace other {

template<class T> static inline T componentwise_min(const T& v1,const T& v2) {
  return T::componentwise_min(v1,v2);
}

static inline float componentwise_min(const float& v1,const float& v2) {
  return min(v1,v2);
}

static inline double componentwise_min(const double& v1,const double& v2) {
  return min(v1,v2);
}

template<class T,class... Rest> static inline T componentwise_min(const T& v1,const T& v2,const Rest&... rest) {
  return componentwise_min(v1,componentwise_min(v2,rest...));
}

template<class T> static inline T componentwise_max(const T& v1,const T& v2) {
  return T::componentwise_max(v1,v2);
}

static inline float componentwise_max(const float& v1,const float& v2) {
  return max(v1,v2);
}

static inline double componentwise_max(const double& v1,const double& v2) {
  return max(v1,v2);
}

template<class T,class... Rest> static inline T componentwise_max(const T& v1,const T& v2,const Rest&... rest) {
  return componentwise_max(v1,componentwise_max(v2,rest...));
}

}
