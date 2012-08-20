//#####################################################################
// Header DataStructuresForward
//#####################################################################
#pragma once

#include <other/core/python/config.h> // Must be included first
#include <boost/mpl/and.hpp>
namespace other{

namespace mpl = boost::mpl;

struct unit{};

template<class... Ts> class Tuple;
template<class TK,class T=unit> class Hashtable;
class UnionFind;

class OperationHash;
template<class T> class Queue;
template<class T> class Stack;

}
