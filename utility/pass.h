//#####################################################################
// Pass utility for variadic templates
//#####################################################################
//
// See http://en.wikipedia.org/wiki/Variadic_template for details.  Example usage:
//
// template<class F,class... Args> void call_many(const F& f, Args&&... args) {
//   OTHER_PASS(f(args)); // Call f on each of the input arguments
// }
//
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
namespace other{

#define OTHER_PASS(expression) {const int _pass_helper[] OTHER_UNUSED = {((expression),1)...};}

}
