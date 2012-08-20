//#####################################################################
// Class HasCheapCopy
//#####################################################################
#pragma once

#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_fundamental.hpp>
namespace other {

namespace mpl = boost::mpl;

template<class T> struct HasCheapCopy:public mpl::or_<boost::is_fundamental<T>,boost::is_enum<T> >{};

}
