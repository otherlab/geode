//#####################################################################
// Class Factorial
//#####################################################################
#pragma once

namespace other {

template<unsigned d> struct Factorial;

template<> struct Factorial<0>{enum Workaround {value=1};};

template<unsigned d> struct Factorial {
  BOOST_STATIC_ASSERT((d<=12));
  enum Workaround {value=d*Factorial<d-1>::value};
};

}
