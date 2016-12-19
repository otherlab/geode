set(module_SRCS
  hash.cpp
  numeric_limits.cpp
  optimal_sort.cpp
  sse.cpp
  uint128.cpp
)

set(module_HEADERS
  argmax.h
  argmin.h
  cbrt.h
  choice.h
  clamp.h
  componentwise.h
  constants.h
  copysign.h
  cube.h
  cyclic_shift.h
  Factorial.h
  fallback_sort.h
  givens.h
  hash.h
  integer_log.h
  inverse.h
  isfinite.h
  isnan.h
  lerp.h
  maxabs.h
  max.h
  maxmag.h
  mean.h
  minabs.h
  min.h
  minmag.h
  One.h
  optimal_sort.h
  popcount.h
  pow.h
  rint.h
  robust.h
  sign.h
  small_sort.h
  sqr.h
  sse.h
  uint128.h
  wrap.h
  Zero.h
)

install_geode_headers(math ${module_HEADERS})

add_geode_module(math ${module_SRCS})
