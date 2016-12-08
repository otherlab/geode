set(module_SRCS
  Action.cpp
  Compute.cpp
  ConstValue.cpp
  Listen.cpp
  Prop.cpp
  PropManager.cpp
  Value.cpp
)

set(module_HEADERS
  Action.h
  Compute.h
  ConstValue.h
  convert.h
  error_value.h
  extract.h
  forward.h
  link.h
  Listen.h
  Prop.h
  PropManager.h
  Value.h
)

install_geode_headers(value ${module_HEADERS})

add_geode_module(value ${module_SRCS})
