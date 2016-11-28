// Platform independent glut include
#pragma once

// Must be included first
#include <geode/python/forward.h>

#ifdef _WIN32
#include <cstdlib>
#define WINDOWS_MEAN_AND_LEAN
#include <windows.h>
// clean up after windows.h
#undef min
#undef max
#undef small
#undef interface
#undef far
#undef near
#endif

#ifdef USE_OSMESA

#include <GL/gl.h>
#include <GL/glext.h>

// GLUT and GLU functions don't work in offscreen rendering mode,
// but are included to avoid splitting all of their dependencies out
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifndef GLAPI
#define GLAPI
#endif
#define GLAPIENTRY
#include <GL/osmesa.h>
#include <gui/osmesa_glew_subs.h>

#else

// only include if not using OSMesa since glew dispatches to incompatable version of OpenGL
#include <GL/glew.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#endif

#ifdef NDEBUG
#define GEODE_CHECK_GL() ((void)0)
#else
#define GEODE_CHECK_GL() \
  do { \
    int error = glGetError(); \
    if (error != GL_NO_ERROR) \
      gl_check_failed(GEODE_DEBUG_FUNCTION_NAME,__FILE__,__LINE__,error); \
  } while (0)
#endif