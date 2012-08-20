//#####################################################################
// Class ExrFile (requires libpng)
//#####################################################################
#ifndef _EXR_FILE_h
#define _EXR_FILE_h

#include <other/core/array/forward.h>
#include <other/core/vector/forward.h>
#include <string>
namespace other{
  
  template<class T>
  class ExrFile
  {
  public:
    ExrFile()
    {}
    
    //#####################################################################
    static Array<Vector<T,3>,2> read(const std::string& filename) OTHER_EXPORT;
    static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image) OTHER_EXPORT;
    static bool is_supported() OTHER_EXPORT;
    //#####################################################################
  };
}
#endif
