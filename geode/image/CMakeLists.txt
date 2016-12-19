set(module_SRCS
  color_utils.cpp
  ExrFile.cpp
  Image.cpp
  JpgFile.cpp
  MovFile.cpp
  PngFile.cpp
)

set(module_HEADERS
  color_utils.h
  ExrFile.h
  forward.h
  Image.h
  JpgFile.h
  MovFile.h
  PngFile.h
)

install_geode_headers(image ${module_HEADERS})

add_geode_module(image ${module_SRCS})
