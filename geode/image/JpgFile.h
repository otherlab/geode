//#####################################################################
// Class JpgFile
//#####################################################################
//
// Requires libjpeg (http://www.ijg.org/)
// Note: Appears to give the error message
//   Jpeg parameter struct mismatch: library thinks size is 372, caller expects 376
// with gcc which goes away if you don't use -malign-double
//#####################################################################
#pragma once

#include <geode/utility/debug.h>
#include <geode/array/forward.h>
#include <geode/vector/forward.h>
#include <string>
namespace geode {

template<class T> class JpgFile {
public:
GEODE_CORE_EXPORT static Array<Vector<T,3>,2> read(const std::string& filename);
GEODE_CORE_EXPORT static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image);
GEODE_CORE_EXPORT static bool is_supported();
};

}
