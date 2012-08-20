#pragma once

#include <other/core/utility/overload.h>
namespace other{

#define OTHER_REMOVE_COMMAS_1(a1)                                                  a1
#define OTHER_REMOVE_COMMAS_2(a1,a2)                                               a1 a2
#define OTHER_REMOVE_COMMAS_3(a1,a2,a3)                                            a1 a2 a3
#define OTHER_REMOVE_COMMAS_4(a1,a2,a3,a4)                                         a1 a2 a3 a4
#define OTHER_REMOVE_COMMAS_5(a1,a2,a3,a4,a5)                                      a1 a2 a3 a4 a5
#define OTHER_REMOVE_COMMAS_6(a1,a2,a3,a4,a5,a6)                                   a1 a2 a3 a4 a5 a6
#define OTHER_REMOVE_COMMAS_7(a1,a2,a3,a4,a5,a6,a7)                                a1 a2 a3 a4 a5 a6 a7
#define OTHER_REMOVE_COMMAS_8(a1,a2,a3,a4,a5,a6,a7,a8)                             a1 a2 a3 a4 a5 a6 a7 a8
#define OTHER_REMOVE_COMMAS_9(a1,a2,a3,a4,a5,a6,a7,a8,a9)                          a1 a2 a3 a4 a5 a6 a7 a8 a9
#define OTHER_REMOVE_COMMAS_10(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)                     a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
#define OTHER_REMOVE_COMMAS_11(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)                 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11
#define OTHER_REMOVE_COMMAS_12(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)             a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12
#define OTHER_REMOVE_COMMAS_13(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13)         a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13
#define OTHER_REMOVE_COMMAS_14(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14)     a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14
#define OTHER_REMOVE_COMMAS_15(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
#define OTHER_REMOVE_COMMAS_16(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16
#define OTHER_REMOVE_COMMAS_17(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17
#define OTHER_REMOVE_COMMAS_18(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18
#define OTHER_REMOVE_COMMAS_19(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19
#define OTHER_REMOVE_COMMAS_20(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20
#define OTHER_REMOVE_COMMAS_21(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21
#define OTHER_REMOVE_COMMAS_22(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22
#define OTHER_REMOVE_COMMAS_23(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23
#define OTHER_REMOVE_COMMAS_24(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24
#define OTHER_REMOVE_COMMAS_25(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25
#define OTHER_REMOVE_COMMAS_26(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26
#define OTHER_REMOVE_COMMAS_27(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27
#define OTHER_REMOVE_COMMAS_28(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28
#define OTHER_REMOVE_COMMAS_29(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29
#define OTHER_REMOVE_COMMAS_30(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a30
#define OTHER_REMOVE_COMMAS_31(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a30 a31
#define OTHER_REMOVE_COMMAS_32(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32) \
  a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a30 a31 a32
#define OTHER_REMOVE_COMMAS(...) OTHER_OVERLOAD(OTHER_REMOVE_COMMAS_,__VA_ARGS__)(__VA_ARGS__)

}
