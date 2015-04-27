// This file provides setup needed to include headers from OpenMesh 
#pragma once
#include <geode/config.h>

// Openmesh uses WIN32 instead of _WIN32 so we provide it
#if defined(GEODE_OPENMESH) && defined(_WIN32) && !defined(WIN32)
  #define WIN32 _WIN32
#endif

#ifdef GEODE_OPENMESH
  #if defined(_WIN32) != defined(WIN32)
    #error OpenMesh uses 'WIN32' macro, which has not been configured to match _WIN32
  #endif

  // For MinGW we have to manually define OM_STATIC_BUILD since OpenMesh config doesn't expect gcc on windows
  #ifdef __MINGW32__
    #define OM_STATIC_BUILD 1
  #endif

  // Since we're using hidden symbol visibility, dynamic_casts across shared library
  // boundaries are problematic. Therefore, don't use them even in debug mode.
  #define OM_FORCE_STATIC_CAST

#endif // GEODE_OPENMESH