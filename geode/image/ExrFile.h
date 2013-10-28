//#####################################################################
// Class ExrFile
//#####################################################################
#pragma once

#include <geode/array/forward.h>
#include <geode/vector/forward.h>
#include <string>
namespace geode {

template<class T> class ExrFile {
public:
GEODE_CORE_EXPORT static Array<Vector<T,3>,2> read(const std::string& filename);
GEODE_CORE_EXPORT static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image);
GEODE_CORE_EXPORT static bool is_supported();
};

}
