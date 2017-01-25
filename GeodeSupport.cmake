include(CheckCXXCompilerFlag)

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
    set(GEODE_MODULE_NAMES ${GEODE_MODULE_NAMES} ${_name} PARENT_SCOPE)

    set_property(
      TARGET ${_name}
      PROPERTY CXX_STANDARD 11
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
        ${GMP_INCLUDE}
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
        -Werror
        -Wno-unused-function
        -Wno-array-bounds
        -Wno-unknown-pragmas
        -Wno-deprecated
        -Wno-format-security
        -Wno-attributes
        -Wno-unused-variable
        -fPIC
    )

    CHECK_CXX_COMPILER_FLAG(-Wno-misleading-indentation COMPILER_CHECKS_MISLEADING_INDENTATION)
    if (COMPILER_CHECKS_MISLEADING_INDENTATION)
      target_compile_options(
        ${_name}
        PUBLIC
          -Wno-misleading-indentation
      )
    endif()


    if (PYTHON_FOUND)
      target_include_directories(
        ${_name}
        PUBLIC
          ${PYTHON_INCLUDE_DIRS}
          ${NUMPY_INCLUDE_DIRS}
      )

      file(GLOB _python_bits *.py)
      set(_pytargets "")
      foreach(_python_bit ${_python_bits})
        get_filename_component(_pyfile ${_python_bit} NAME)
        add_custom_command(
          OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_pyfile}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${_pyfile}" "${CMAKE_CURRENT_BINARY_DIR}/${_pyfile}"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_pyfile}"
          COMMENT "Copying ${_pyfile} to build tree"
        )
      list(APPEND _pytargets "${CMAKE_CURRENT_BINARY_DIR}/${_pyfile}")
      endforeach()
      add_custom_target(${_name}-python
        DEPENDS ${_pytargets}
      )
      add_dependencies(${_name} ${_name}-python)
    endif()
  endif()
endmacro(ADD_GEODE_MODULE)

function(STATUS _name _value)
  list(APPEND STATUS_TABLE "${_name}")
  list(APPEND STATUS_TABLE "${_value}")
  set(STATUS_TABLE "${STATUS_TABLE}" PARENT_SCOPE)
endfunction(STATUS)

function(STATUS_REPORT)
  message(STATUS "Build configuration:")
  set(_isName TRUE)
  set(_max_name_width 0)
  foreach(v IN LISTS STATUS_TABLE)
    if (_isName)
      set(_isName FALSE)
      string(LENGTH "${v}" _name_length)
      if (_name_length GREATER _max_name_width)
        set(_max_name_width ${_name_length})
      endif()
    else()
      set(_isName TRUE)
    endif()
  endforeach()
  while(STATUS_TABLE)
    list(GET STATUS_TABLE 0 _name)
    list(GET STATUS_TABLE 1 _value)
    list(REMOVE_AT STATUS_TABLE 0 1)

    string(LENGTH "${_name}" _name_length)
    math(EXPR _padding_size "${_max_name_width}-${_name_length} + 1")
    string(RANDOM LENGTH ${_padding_size} ALPHABET " " _padding)

    message(STATUS "\t${_name}${_padding}\t${_value}")
  endwhile()
endfunction()
