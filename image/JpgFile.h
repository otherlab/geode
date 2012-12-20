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

#include <other/core/utility/debug.h>
#include <other/core/array/forward.h>
#include <other/core/vector/forward.h>
#include <string>
namespace other {

template<class T> class JpgFile {
public:
OTHER_CORE_EXPORT static Array<Vector<T,3>,2> read(const std::string& filename);
OTHER_CORE_EXPORT static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image);
OTHER_CORE_EXPORT static bool is_supported();
};

}
