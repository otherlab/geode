#pragma once

#include <boost/mpl/bool.hpp>
namespace geode {

namespace mpl = boost::mpl;

class ValueBase;
class PropBase;
template<class T> class Value;
template<class T> class ValueRef;
class Action;
template<class T> class Prop;
template<class T> class PropRef;
class Listen;
class PropManager;

template<class T> struct has_clamp : public mpl::false_{};
template<> struct has_clamp<int> : public mpl::true_{};
template<> struct has_clamp<double> : public mpl::true_{};

}
