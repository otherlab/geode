//#####################################################################
// Class ExrFile
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/vector/forward.h>
#include <string>
namespace other {

template<class T> class ExrFile {
public:
  static Array<Vector<T,3>,2> read(const std::string& filename) OTHER_CORE_EXPORT;
  static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image) OTHER_CORE_EXPORT;
  static bool is_supported() OTHER_CORE_EXPORT;
};

}
