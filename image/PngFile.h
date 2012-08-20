//#####################################################################
// Class PngFile (requires libpng)
//#####################################################################
#ifndef _PNG_FILE_h
#define _PNG_FILE_h

#include <string>
#include <vector>
#include <other/core/array/forward.h>
#include <other/core/vector/forward.h>
namespace other{

template<class T>
class PngFile
{
public:
    PngFile()
    {}

//#####################################################################
    static Array<Vector<T,3>,2> read(const std::string& filename) OTHER_EXPORT;
    static Array<Vector<T,4>,2> read_alpha(const std::string& filename) OTHER_EXPORT;
    static void write(const std::string& filename,RawArray<const Vector<T,3>,2> image) OTHER_EXPORT;
    static void write(const std::string& filename,RawArray<const Vector<T,4>,2> image) OTHER_EXPORT;
    static std::vector<unsigned char> write_to_memory(RawArray<const Vector<T,3>,2> image) OTHER_EXPORT;
    static std::vector<unsigned char> write_to_memory(RawArray<const Vector<T,4>,2> image) OTHER_EXPORT;
    static bool is_supported() OTHER_EXPORT;
//#####################################################################
};
}
#endif
