// Locate extra resource files (textures, etc.) at runtime
#pragma once

/* To use, install the resources in an SConscript via something like
 *
 *   resource(env,"images")
 *
 * and look up individual files at runtime via something like
 *
 *   string path = resource("images","metal.jpg");
 */

#include <geode/utility/path.h>
namespace geode {

using std::string;

// Get the base path for resources
GEODE_CORE_EXPORT string resource_path();

// Get the path to the given resource.  Alias for path::join(resource_path(),path).
GEODE_CORE_EXPORT string resource(const string& path);
GEODE_CORE_EXPORT string resource(const string& path0, const string& path1);
GEODE_CORE_EXPORT string resource(const string& path0, const string& path1, const string& path2);

}
