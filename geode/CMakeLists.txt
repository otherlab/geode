option(GEODE_THREAD_SAFE "Compile with thread safety" TRUE)

if (PYTHON_FOUND AND NOT GEODE_DISABLE_PYTHON)
  set(GEODE_PYTHON YES)
endif()
if (GMP_FOUND)
  set(GEODE_GMP YES)
endif()
if (JPEG_FOUND)
  set(GEODE_LIBJPEG YES)
endif()
if (PNG_FOUND)
  set(GEODE_LIBPNG YES)
endif()

# Check Python binding configuration
if (GEODE_REQUIRE_PYTHON AND NOT GEODE_PYTHON)
  if(NOT PYTHON_FOUND)
    message(WARNING "Python libs not found.")
  endif()
  message(FATAL_ERROR "Python bindings could not be configured as required")
endif()

if(GEODE_PYTHON)
  message(STATUS "Python bindings for geode enabled")
else()
  message(STATUS "Python bindings for geode disabled")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION include/geode/)

set(GEODE_MODULE_OBJECTS "" PARENT_SCOPE)

add_subdirectory(array)
add_subdirectory(python)
add_subdirectory(force)
add_subdirectory(utility)
add_subdirectory(math)
add_subdirectory(geometry)
add_subdirectory(openmesh)
add_subdirectory(random)
add_subdirectory(structure)
add_subdirectory(solver)
add_subdirectory(image)
add_subdirectory(svg)
add_subdirectory(mesh)
add_subdirectory(value)
add_subdirectory(vector)

if (GMP_FOUND)
  add_subdirectory(exact)
endif()

add_library(geode SHARED ${GEODE_MODULE_OBJECTS})
target_include_directories(
  geode
  INTERFACE
    $<INSTALL_INTERFACE:include/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
)

if (GEODE_PYTHON)
  target_include_directories(
    geode
    INTERFACE
      ${PYTHON_INCLUDE_DIRS}
      ${NUMPY_INCLUDE_DIRS}
  )
endif()

set_target_properties(geode PROPERTIES BUILD_WITH_INSTALL_RPATH 1 INSTALL_NAME_DIR "${CMAKE_CURRENT_BINARY_DIR}")

install(TARGETS geode EXPORT Geode DESTINATION lib)
install(EXPORT Geode FILE GeodeConfig.cmake DESTINATION cmake)
export(TARGETS geode FILE ${CMAKE_CURRENT_BINARY_DIR}/GeodeConfig.cmake)

if (GMP_FOUND)
  target_include_directories(
    geode
    PUBLIC
      ${GMP_INCLUDE}
  )

  target_link_libraries(
    geode
    PUBLIC
      ${GMP_LIBRARIES}
  )
else()
  message(STATUS "GMP_FOUND not set")
endif()

if (OPENMESH_FOUND)
  target_include_directories(
    geode
    PUBLIC
      ${OPENMESH_INCLUDE}
  )

  target_link_libraries(
    geode
    PUBLIC
      ${OPENMESH_LIBRARIES}
  )
else()
  message(STATUS "OPENMESH_FOUND not set")
endif()

if (JPEG_FOUND)
  target_include_directories(
    geode
    PUBLIC
      ${JPEG_INCLUDE}
  )

  target_link_libraries(
    geode
    PUBLIC
      ${JPEG_LIBRARIES}
  )
endif()

if (PNG_FOUND)
  target_include_directories(
    geode
    PUBLIC
      ${PNG_INCLUDE}
  )

  target_link_libraries(
    geode
    PUBLIC
      ${PNG_LIBRARIES}
  )
endif()

if (GEODE_PYTHON)
  target_link_libraries(
    geode
    PUBLIC
      ${PYTHON_LIBRARIES}
  )

  add_python_module(geode_wrap module.cpp)
  target_link_libraries(
    geode_wrap
    PUBLIC
      geode
  )

  target_include_directories(
    geode_wrap
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../
      ${CMAKE_CURRENT_BINARY_DIR}/../
  )

  target_compile_features(
    geode_wrap
    PUBLIC
      cxx_static_assert
  )

  file(COPY __init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  install(CODE "execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")

  add_custom_target(develop DEPENDS geode)
  add_custom_command(
    TARGET develop
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/setup.py develop
  )
  install(CODE "execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")
endif()
