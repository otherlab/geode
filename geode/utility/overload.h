//#####################################################################
// Macro GEODE_OVERLOAD
//#####################################################################
#pragma once

#include <geode/utility/config.h>
namespace geode {

// See http://groups.google.com/group/comp.std.c/browse_thread/thread/7cdd9f2984c15e3e/effa7cda7c378dac%23effa7cda7c378dac for macro explanations

#define GEODE_CAT(a,...) GEODE_PRIMITIVE_CAT(a,__VA_ARGS__)
#define GEODE_PRIMITIVE_CAT(a,...) a ## __VA_ARGS__
#define GEODE_SPLIT(i,...) GEODE_PRIMITIVE_CAT(GEODE_SPLIT_,i)(__VA_ARGS__)
#define GEODE_SPLIT_0(a,...) a
#define GEODE_SPLIT_1(a,...) __VA_ARGS__
#define GEODE_COMMA() ,
#define GEODE_REM(...) __VA_ARGS__
#define GEODE_SIZE(...) GEODE_SPLIT(0,GEODE_SPLIT(1,GEODE_SIZE_A(GEODE_COMMA,GEODE_REM(__VA_ARGS__)),,))
#define GEODE_SIZE_A(_,im) GEODE_SIZE_B(im,_()32,_()31,_()30,_()29,_()28,_()27,_()26,_()25,_()24,_()23,_()22,_()21,_()20,_()19,_()18,_()17,_()16,_()15,_()14,_()13,_()12,_()11,_()10,_()9,_()8,_()7,_()6,_()5,_()4,_()3,_()2,_()1)
#define GEODE_SIZE_B(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,_,...) _

#define GEODE_OVERLOAD(prefix,...) GEODE_CAT(prefix,GEODE_SIZE(__VA_ARGS__))

#define GEODE_ONE_OR_MORE(...) GEODE_SPLIT(0,GEODE_SPLIT(1,GEODE_ONE_OR_MORE_A(GEODE_COMMA,GEODE_REM(__VA_ARGS__)),,))
#define GEODE_ONE_OR_MORE_A(_,im) GEODE_ONE_OR_MORE_B(im, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()ONE)
#define GEODE_ONE_OR_MORE_B(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,_,...) _
#define GEODE_OVERLOAD_ONE_OR_MORE(prefix,...) GEODE_CAT(prefix,GEODE_ONE_OR_MORE(__VA_ARGS__))

}
