//#####################################################################
// Class HasCheapCopy
//#####################################################################
#pragma once

#include <geode/utility/type_traits.h>
#include <boost/mpl/or.hpp>
namespace geode {

namespace mpl = boost::mpl;

template<class T> struct HasCheapCopy:public mpl::or_<is_fundamental<T>,is_enum<T>>{};

}
