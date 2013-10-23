//#####################################################################
// Class CopyConst
//#####################################################################
#pragma once

#include <boost/type_traits/add_const.hpp>
namespace geode {

namespace mpl = boost::mpl;

template<class T,class S> struct CopyConst{typedef T type;};
template<class T,class S> struct CopyConst<T,const S>:public boost::add_const<T>{};

}
