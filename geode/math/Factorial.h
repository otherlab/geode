//#####################################################################
// Class Factorial
//#####################################################################
#pragma once

namespace geode {

template<unsigned d> struct Factorial;

template<> struct Factorial<0>{enum Workaround {value=1};};

template<unsigned d> struct Factorial {
  static_assert(d<=12,"d! would overflow int");
  enum Workaround {value=d*Factorial<d-1>::value};
};

}
