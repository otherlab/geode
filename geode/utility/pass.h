//#####################################################################
// Pass utility for variadic templates
//#####################################################################
//
// See http://en.wikipedia.org/wiki/Variadic_template for details.  Example usage:
//
// template<class F,class... Args> void call_many(const F& f, Args&&... args) {
//   GEODE_PASS(f(args)); // Call f on each of the input arguments
// }
//
//#####################################################################
#pragma once

#include <geode/utility/config.h>
namespace geode {

#define GEODE_PASS(expression) {const int _pass_helper[] GEODE_UNUSED = {((expression),1)...};}

}
