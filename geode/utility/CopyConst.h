//#####################################################################
// Class CopyConst
//#####################################################################
#pragma once

#include <geode/utility/type_traits.h>
namespace geode {

template<class T,class S> struct CopyConst{typedef T type;};
template<class T,class S> struct CopyConst<T,const S>:public add_const<T>{};

}
