//#####################################################################
// Function mean
//#####################################################################
#pragma once

#include <geode/vector/ScalarPolicy.h>
namespace geode {

template<class TV> inline const TV& mean(const TV& x) {
  return x;
}

template<class TV> inline TV mean(const TV& x, const TV& y) {
  return typename ScalarPolicy<TV>::type(.5)*(x+y);
}

template<class TV> inline TV mean(const TV& x, const TV& y, const TV& z) {
  return typename ScalarPolicy<TV>::type(1./3)*(x+y+z);
}

template<class TV> inline TV mean(const TV& x, const TV& y, const TV& z, const TV& w) {
  return typename ScalarPolicy<TV>::type(.25)*(x+y+z+w);
}

}
