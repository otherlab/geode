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
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
        ${CMAKE_CURRENT_BINARY_DIR}/../../
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

    CHECK_CXX_COMPILER_FLAG(-Wno-undefined-var-template COMPILER_CHECKS_UNDEFINED_VAR_TEMPLATE)
    if (COMPILER_CHECKS_UNDEFINED_VAR_TEMPLATE)
      target_compile_options(
        ${_name}
        PUBLIC
          -Wno-undefined-var-template
      )
    endif()

    CHECK_CXX_COMPILER_FLAG(-Wno-misleading-indentation COMPILER_CHECKS_MISLEADING_INDENTATION)
    if (COMPILER_CHECKS_MISLEADING_INDENTATION)
      target_compile_options(
        ${_name}
        PUBLIC
          -Wno-misleading-indentation
      )
    endif()


    if (GEODE_PYTHON)
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
        ALL DEPENDS ${_pytargets}
      )
    endif()
  endif()
endmacro(ADD_GEODE_MODULE)
