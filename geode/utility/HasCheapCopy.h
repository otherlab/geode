//#####################################################################
// Class HasCheapCopy
//#####################################################################
#pragma once

#include <geode/utility/type_traits.h>
namespace geode {

template<class T> struct HasCheapCopy:public mpl::or_<is_fundamental<T>,is_enum<T>>{};

}
