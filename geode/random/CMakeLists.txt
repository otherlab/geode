set(module_SRCS
  counter.cpp
  permute.cpp
  Random.cpp
  Sobol.cpp
)

set(module_HEADERS
  counter.h
  forward.h
  permute.h
  Random.h
  Sobol.h
)

install_geode_headers(random ${module_HEADERS})

add_geode_module(random ${module_SRCS})
