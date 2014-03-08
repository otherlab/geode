// Optimal sorting networks up to n = 10
//
// For details, see
//   http://en.wikipedia.org/wiki/Sorting_network
//   Knuth. The Art of Computer Programming, Volume 3: Sorting and Searching, 1997.
#pragma once

// C(i,j) is a comparator on indices i,j, and L() starts a level.
#define GEODE_SORT_NETWORK(n,C,L) GEODE_SORT_NETWORK##n(C,L) 

#define GEODE_SORT_NETWORK0(C,L)
#define GEODE_SORT_NETWORK1(C,L)
#define GEODE_SORT_NETWORK2(C,L) \
  L() C(0,1)
#define GEODE_SORT_NETWORK3(C,L) \
  L() C(0,1) \
  L() C(1,2) \
  L() C(0,1)
#define GEODE_SORT_NETWORK4(C,L) \
  L() C(0,1) C(2,3) \
  L() C(0,2) C(1,3) \
  L() C(1,2)
#define GEODE_SORT_NETWORK5(C,L) \
  L() C(1,2) C(3,4) \
  L() C(1,3) C(0,2) \
  L() C(0,3) C(2,4) \
  L() C(0,1) C(2,3) \
  L() C(1,2)
#define GEODE_SORT_NETWORK6(C,L) \
  L() C(0,1) C(2,3) C(4,5) \
  L() C(0,2) C(3,5) C(1,4) \
  L() C(0,1) C(2,3) C(4,5) \
  L() C(1,2) C(3,4) \
  L() C(2,3)
#define GEODE_SORT_NETWORK7(C,L) \
  L() C(1,2) C(3,4) C(5,6) \
  L() C(0,2) C(3,5) C(4,6) \
  L() C(0,4) C(1,5) C(2,6) \
  L() C(0,3) C(2,5) \
  L() C(1,3) C(2,4) \
  L() C(0,1) C(2,3) C(4,5)
#define GEODE_SORT_NETWORK8(C,L) \
  L() C(0,1) C(2,3) C(4,5) C(6,7) \
  L() C(0,2) C(1,3) C(4,6) C(5,7) \
  L() C(0,4) C(1,2) C(3,7) C(5,6) \
  L() C(1,5) C(2,6) \
  L() C(2,4) C(3,5) \
  L() C(1,2) C(3,4) C(5,6)
#define GEODE_SORT_NETWORK9(C,L) \
  L() C(1,8) C(2,7) C(3,6) C(4,5) \
  L() C(0,2) C(1,4) C(5,8) C(6,7) \
  L() C(0,3) C(2,6) C(4,5) C(7,8) \
  L() C(0,1) C(2,4) C(3,5) C(6,7) \
  L() C(1,3) C(4,6) C(5,7) \
  L() C(1,2) C(3,4) C(5,6) C(7,8) \
  L() C(2,3) C(4,5)
#define GEODE_SORT_NETWORK10(C,L) \
  L() C(0,1) C(2,3) C(4,5) C(6,7) C(8,9) \
  L() C(0,5) C(1,8) C(2,6) C(3,7) C(4,9) \
  L() C(0,2) C(1,4) C(3,6) C(5,8) C(7,9) \
  L() C(0,1) C(2,7) C(3,5) C(4,6) C(8,9) \
  L() C(1,3) C(2,4) C(5,7) C(6,8) \
  L() C(1,2) C(3,4) C(5,6) C(7,8) \
  L() C(2,3) C(4,5) C(6,7)
// If you add to this list, modify the unit test in the .cpp appropriately.
