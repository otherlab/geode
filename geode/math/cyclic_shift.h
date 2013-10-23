//#####################################################################
// Cyclic shifts
//#####################################################################
//
// Cyclic shift 3 or 4 values
//
//#####################################################################
#pragma once

namespace geode {

template<class T> inline void cyclic_shift(T& i, T& j, T& k) {
  T temp=k;k=j;j=i;i=temp;
}

template<class T> inline void cyclic_shift(T& i, T& j, T& k, T& l) {
  T temp=l;l=k;k=j;j=i;i=temp;
}

}
