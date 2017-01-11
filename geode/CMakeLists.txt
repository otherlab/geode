option(GEODE_THREAD_SAFE "Compile with thread safety" TRUE)
if (PYTHON_FOUND)
  set(GEODE_PYTHON YES)
endif()
if (GMP_FOUND)
  set(GEODE_GMP YES)
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

if (PYTHON_FOUND)
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
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}
  )

  target_compile_features(
    geode_wrap
    PUBLIC
      cxx_static_assert
  )

  file(COPY __init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  install(CODE "execute_process(COMMAND ${CMAKE_SOURCE_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")

  add_custom_target(develop DEPENDS geode)
  add_custom_command(
    TARGET develop
    COMMAND ${CMAKE_SOURCE_DIR}/setup.py develop
  )
  install(CODE "execute_process(COMMAND ${CMAKE_SOURCE_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")
endif()
