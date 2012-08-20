//#####################################################################
// Macro OTHER_OVERLOAD
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
namespace other {

// See http://groups.google.com/group/comp.std.c/browse_thread/thread/7cdd9f2984c15e3e/effa7cda7c378dac%23effa7cda7c378dac for macro explanations

#define OTHER_CAT(a,...) OTHER_PRIMITIVE_CAT(a,__VA_ARGS__)
#define OTHER_PRIMITIVE_CAT(a,...) a ## __VA_ARGS__
#define OTHER_SPLIT(i,...) OTHER_PRIMITIVE_CAT(OTHER_SPLIT_,i)(__VA_ARGS__)
#define OTHER_SPLIT_0(a,...) a
#define OTHER_SPLIT_1(a,...) __VA_ARGS__
#define OTHER_COMMA() ,
#define OTHER_REM(...) __VA_ARGS__
#define OTHER_SIZE(...) OTHER_SPLIT(0,OTHER_SPLIT(1,OTHER_SIZE_A(OTHER_COMMA,OTHER_REM(__VA_ARGS__)),,))
#define OTHER_SIZE_A(_,im) OTHER_SIZE_B(im,_()32,_()31,_()30,_()29,_()28,_()27,_()26,_()25,_()24,_()23,_()22,_()21,_()20,_()19,_()18,_()17,_()16,_()15,_()14,_()13,_()12,_()11,_()10,_()9,_()8,_()7,_()6,_()5,_()4,_()3,_()2,_()1)
#define OTHER_SIZE_B(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,_,...) _

#define OTHER_OVERLOAD(prefix,...) OTHER_CAT(prefix,OTHER_SIZE(__VA_ARGS__))

#define OTHER_ONE_OR_MORE(...) OTHER_SPLIT(0,OTHER_SPLIT(1,OTHER_ONE_OR_MORE_A(OTHER_COMMA,OTHER_REM(__VA_ARGS__)),,))
#define OTHER_ONE_OR_MORE_A(_,im) OTHER_ONE_OR_MORE_B(im, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()MORE, _()ONE)
#define OTHER_ONE_OR_MORE_B(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,_,...) _
#define OTHER_OVERLOAD_ONE_OR_MORE(prefix,...) OTHER_CAT(prefix,OTHER_ONE_OR_MORE(__VA_ARGS__))

}
