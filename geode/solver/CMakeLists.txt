set(module_SRCS
  brent.cpp
  pattern_max.cpp
  powell.cpp
)

set(module_HEADERS
  brent.h
  pattern_max.h
  powell.h
  quadratic.h
)

install_geode_headers(solver ${module_HEADERS})

add_geode_module(solver ${module_SRCS})
