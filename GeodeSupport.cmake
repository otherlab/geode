function(add_python_module _name)
  add_library(
    ${_name} MODULE ${ARGN}
  )

  target_include_directories(
    ${_name}
    PUBLIC
      ${PYTHON_INCLUDE_DIRS}
      ${NUMPY_INCLUDE_DIRS}
      ${CMAKE_BINARY_DIR}
  )

  target_link_libraries(
    ${_name}
    PUBLIC
      ${PYTHON_LIBRARIES}
  )

  set_target_properties(
    ${_name}
    PROPERTIES
      PREFIX ""
  )
endfunction()

set(GEODE_MODULES)

macro(INSTALL_GEODE_HEADERS _name)
  install(FILES ${ARGN} DESTINATION include/geode/${_name}/)
endmacro(INSTALL_GEODE_HEADERS)

macro(ADD_GEODE_MODULE _name)
  option(BUILD_GEODE_${_name} "Build the ${_name} module" TRUE)
  if (BUILD_GEODE_${_name})
    message(STATUS "Building ${_name} module.")

    # Handle the NO_MODULE argument
    # If it exists in the source list, remove it and don't add module.cpp
    # If it doesn't, do nothing and add module.cpp
    set(_srcs ${ARGN})
    list(FIND _srcs NO_MODULE _no_module)

    if (_no_module EQUAL -1)
      list(APPEND _srcs module.cpp)
    else()
      list(REMOVE_AT _srcs ${_no_module})
    endif()

    add_library(
      ${_name} OBJECT ${_srcs}
    )

    set(GEODE_MODULE_OBJECTS ${GEODE_MODULE_OBJECTS} $<TARGET_OBJECTS:${_name}> PARENT_SCOPE)

    target_compile_features(
      ${_name}
      PUBLIC
        cxx_static_assert
    )

    target_compile_definitions(
      ${_name}
      PRIVATE
        BUILDING_geode
    )

    target_include_directories(
      ${_name}
      PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    
    if (GMP_FOUND)
      target_include_directories(
        ${_name}
        PUBLIC
        ${GMP_INCLUDE_DIR}
      )
    endif()

    target_compile_options(
      ${_name}
      PUBLIC
        -march=native
        -mtune=native
        -O3
        -funroll-loops
        -Wall
        -Winit-self
        -Woverloaded-virtual
        -Wsign-compare
        -fno-strict-aliasing
        -std=c++0x
        -Werror
        -Wno-unused-function
        -Wno-misleading-indentation
        -Wno-array-bounds
        -Wno-unknown-pragmas
        -Wno-deprecated
        -Wno-unused-variable
        -Wno-format-security
        -Wno-attributes
        -fPIC
    )

    if (PYTHON_FOUND)
      target_include_directories(
        ${_name}
        PUBLIC
          ${PYTHON_INCLUDE_DIRS}
          ${NUMPY_INCLUDE_DIRS}
      )

      file(GLOB _python_bits *.py)
      file(COPY ${_python_bits} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  endif()
endmacro(ADD_GEODE_MODULE)
