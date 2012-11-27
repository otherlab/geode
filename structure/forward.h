//#####################################################################
// Header DataStructuresForward
//#####################################################################
#pragma once

#include <other/core/utility/config.h> // Must be included first
#include <boost/mpl/and.hpp>
namespace other{

namespace mpl = boost::mpl;

struct unit{};

#ifdef OTHER_VARIADIC
template<class... Ts> class Tuple;
#else
template<class T0=void,class T1=void,class T2=void,class T3=void,class T4=void> class Tuple;
#endif

class UnionFind;

class OperationHash;
template<class T> class Queue;
template<class T> class Stack;

template<class TK,class T> class HashtableEntry;
template<class TK,class T=unit> class Hashtable;

}
