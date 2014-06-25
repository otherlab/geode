//#####################################################################
// Header DataStructuresForward
//#####################################################################
#pragma once

#include <geode/utility/config.h> // Must be included first
namespace geode {

// Convenience empty class
struct Unit {};
static const Unit unit = Unit();

#ifdef GEODE_VARIADIC
template<class... Ts> class Tuple;
#else
template<class T0=void,class T1=void,class T2=void,class T3=void,class T4=void,class T5=void,class T6=void> class Tuple;
#endif

class UnionFind;

class OperationHash;
template<class T> class Queue;
template<class T> class Stack;

template<class TK,class T> struct HashtableEntry;
template<class TK,class T=Unit> class Hashtable;

}
