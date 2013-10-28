//#####################################################################
// Class PngFile (requires libpng)
//#####################################################################
#pragma once

#include <string>
#include <vector>
#include <geode/array/forward.h>
#include <geode/vector/forward.h>
namespace geode {

template<class T> class PngFile {
public:
GEODE_CORE_EXPORT static Array<Vector<T,3>,2> read(const std::string& filename);
GEODE_CORE_EXPORT static Array<Vector<T,4>,2> read_alpha(const std::string& filename);
GEODE_CORE_EXPORT static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image);
GEODE_CORE_EXPORT static void write(const std::string& filename,RawArray<const Vector<T,4>,2> image);
GEODE_CORE_EXPORT static std::vector<unsigned char> write_to_memory(RawArray<const Vector<T,3>,2> image);
GEODE_CORE_EXPORT static std::vector<unsigned char> write_to_memory(RawArray<const Vector<T,4>,2> image);
GEODE_CORE_EXPORT static bool is_supported();
};

}
