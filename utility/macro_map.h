#pragma once

#include <other/core/utility/overload.h>
namespace other {

#define OTHER_MAP_1(f,a1)                                                  f(a1)
#define OTHER_MAP_2(f,a1,a2)                                               f(a1),f(a2)
#define OTHER_MAP_3(f,a1,a2,a3)                                            f(a1),f(a2),f(a3)
#define OTHER_MAP_4(f,a1,a2,a3,a4)                                         f(a1),f(a2),f(a3),f(a4)
#define OTHER_MAP_5(f,a1,a2,a3,a4,a5)                                      f(a1),f(a2),f(a3),f(a4),f(a5)
#define OTHER_MAP_6(f,a1,a2,a3,a4,a5,a6)                                   f(a1),f(a2),f(a3),f(a4),f(a5),f(a6)
#define OTHER_MAP_7(f,a1,a2,a3,a4,a5,a6,a7)                                f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7)
#define OTHER_MAP_8(f,a1,a2,a3,a4,a5,a6,a7,a8)                             f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8)
#define OTHER_MAP_9(f,a1,a2,a3,a4,a5,a6,a7,a8,a9)                          f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9)
#define OTHER_MAP_10(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)                     f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10)
#define OTHER_MAP_11(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)                 f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11)
#define OTHER_MAP_12(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12)             f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12)
#define OTHER_MAP_13(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13)         f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13)
#define OTHER_MAP_14(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14)     f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14)
#define OTHER_MAP_15(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15)
#define OTHER_MAP_16(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16)
#define OTHER_MAP_17(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17)
#define OTHER_MAP_18(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18)
#define OTHER_MAP_19(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19)
#define OTHER_MAP_20(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20)
#define OTHER_MAP_21(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20), f(a21)
#define OTHER_MAP_22(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22)
#define OTHER_MAP_23(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23)
#define OTHER_MAP_24(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24)
#define OTHER_MAP_25(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25)
#define OTHER_MAP_26(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26)
#define OTHER_MAP_27(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27)
#define OTHER_MAP_28(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27),f(a28)
#define OTHER_MAP_29(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27),f(a28),f(a29)
#define OTHER_MAP_30(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27),f(a28),f(a29),f(a30)
#define OTHER_MAP_31(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27),f(a28),f(a29),f(a30),f(a31)
#define OTHER_MAP_32(f,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32) \
  f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8),f(a9),f(a10),f(a11),f(a12),f(a13),f(a14),f(a15),f(a16),f(a17),f(a18),f(a19),f(a20),f(a21),f(a22),f(a23),f(a24),f(a25),f(a26),f(a27),f(a28),f(a29),f(a30),f(a31),f(a32)
#define OTHER_MAP(f,...) OTHER_OVERLOAD(OTHER_MAP_,__VA_ARGS__)(f,__VA_ARGS__)

}
