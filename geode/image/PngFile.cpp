//#####################################################################
// Class PngFile
//#####################################################################
#ifdef _WIN32
// libpng uses far
#define LEAVE_WINDOWS_DEFINES_ALONE
#endif

#include <geode/python/config.h> // Must be included first
#ifdef GEODE_LIBPNG
#define PNG_SKIP_SETJMP_CHECK // Both png and python want to be included first
#include <png.h>
#endif
#include <geode/image/PngFile.h>
#include <geode/image/Image.h>
#include <geode/array/Array2d.h>
#include <geode/vector/Vector3d.h>
#include <geode/utility/Log.h>
namespace geode {
//#####################################################################
// Read/Write stubs for case of no libpng
//#####################################################################
#ifndef GEODE_LIBPNG

template<class T> Array<Vector<T,3>,2> PngFile<T>::read(const std::string& filename) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot read png image.");
}

template<class T> Array<Vector<T,4>,2> PngFile<T>::read_alpha(const std::string& filename) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot read png image.");
}

template<class T> void PngFile<T>::write(const std::string& filename,RawArray<const Vector<T,3>,2> image) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot write png image.");
}

template<class T> void PngFile<T>::write(const std::string& filename,RawArray<const Vector<T,4>,2> image) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot write png image.");
}

template<class T> vector<uint8_t> PngFile<T>::write_to_memory(RawArray<const Vector<T,3>,2> image) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot write png image.");
}

template<class T> vector<uint8_t> PngFile<T>::write_to_memory(RawArray<const Vector<T,4>,2> image) {
  GEODE_FATAL_ERROR("Not compiled with GEODE_LIBPNG.  Cannot write png image.");
}

template<class T> bool PngFile<T>::is_supported() {
  return false;
}

#else
//#####################################################################
// Function Read
//#####################################################################
template<class T> Array<Vector<T,3>,2> PngFile<T>::
read(const std::string& filename)
{
    FILE* file=fopen(filename.c_str(),"rb");
    if(!file) throw IOError(format("Failed to open %s for reading",filename));

    png_structp png_ptr=png_create_read_struct(PNG_LIBPNG_VER_STRING,0,0,0);
    if(!png_ptr) throw IOError(format("Error reading png file %s",filename));
    png_infop info_ptr=png_create_info_struct(png_ptr);
    if(!info_ptr) throw IOError(format("Error reading png file %s",filename));
    if(setjmp(png_jmpbuf(png_ptr))) throw IOError(format("Error reading png file %s",filename));
    png_init_io(png_ptr,file);
    png_read_png(png_ptr,info_ptr,PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND,0);
    int width=png_get_image_width(png_ptr,info_ptr),height=png_get_image_height(png_ptr,info_ptr);

    Array<Vector<T,3>,2> image(height,width);
    Vector<unsigned char,3>** row_pointers=(Vector<unsigned char,3>**)png_get_rows(png_ptr,info_ptr);
    for(int i=0;i<width;i++)for(int j=0;j<height;j++) image(j,i)=Image<T>::to_scalar_color(row_pointers[height-j-1][i]);

    png_destroy_read_struct(&png_ptr,&info_ptr,0);
    fclose(file);
    return image;
}

template<class T> Array<Vector<T,4>,2> PngFile<T>::
read_alpha(const std::string& filename)
{
    FILE* file=fopen(filename.c_str(),"rb");
    if(!file) throw IOError(format("Failed to open %s for reading",filename));

    png_structp png_ptr=png_create_read_struct(PNG_LIBPNG_VER_STRING,0,0,0);
    if(!png_ptr) throw IOError(format("Error reading png file %s",filename));
    png_infop info_ptr=png_create_info_struct(png_ptr);
    if(!info_ptr) throw IOError(format("Error reading png file %s",filename));
    if(setjmp(png_jmpbuf(png_ptr))) throw IOError(format("Error reading png file %s",filename));
    png_init_io(png_ptr,file);
    png_read_png(png_ptr,info_ptr,PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND,0);
    int width=png_get_image_width(png_ptr,info_ptr),height=png_get_image_height(png_ptr,info_ptr);

    Array<Vector<T,4>,2> image(height,width);
    Vector<unsigned char,4>** row_pointers=(Vector<unsigned char,4>**)png_get_rows(png_ptr,info_ptr);

    for(int i=0;i<width;i++){
      for(int j=0;j<height;j++) {
        Vector<unsigned char,4>& v = row_pointers[height-j-1][i];
        image(j,i)=Image<T>::to_scalar_color(v);
      }
    }

    png_destroy_read_struct(&png_ptr,&info_ptr,0);
    fclose(file);
    return image;
}
//#####################################################################
// Function Write
//#####################################################################
template<int C> struct ColorPolicy {};

template<> struct ColorPolicy<3> {
  enum {Type = PNG_COLOR_TYPE_RGB};
};
template<> struct ColorPolicy<4> {
  enum {Type = PNG_COLOR_TYPE_RGBA};
};

template<class T, int C> void
write_helper(const std::string& filename,RawArray<const Vector<T,C>,2> image)
{
    FILE* file=fopen(filename.c_str(),"wb");
    if(!file) GEODE_FATAL_ERROR(format("Failed to open %s for writing",filename));

    png_structp png_ptr=png_create_write_struct(PNG_LIBPNG_VER_STRING,0,0,0);
    if(!png_ptr) GEODE_FATAL_ERROR(format("Error writing png file %s",filename));
    png_infop info_ptr=png_create_info_struct(png_ptr);
    if(!info_ptr) GEODE_FATAL_ERROR(format("Error writing png file %s",filename));
    if(setjmp(png_jmpbuf(png_ptr))) GEODE_FATAL_ERROR(format("Error writing png file %s",filename));
    png_init_io(png_ptr,file);
    png_set_IHDR(png_ptr,info_ptr,image.m,image.n,8,ColorPolicy<C>::Type,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);

    Vector<unsigned char,C>* byte_data=new Vector<unsigned char,C>[image.n*image.m];
    Vector<unsigned char,C>** row_pointers=new Vector<unsigned char,C>*[image.n];
    for(int j=0;j<image.n;j++){
        row_pointers[image.n-j-1]=byte_data+image.m*(image.n-j-1);
        for(int i=0;i<image.m;i++) row_pointers[image.n-j-1][i]=Image<T>::to_byte_color(image(i,j));}
    png_set_rows(png_ptr,info_ptr,(png_byte**)row_pointers);
    png_write_png(png_ptr,info_ptr,PNG_TRANSFORM_IDENTITY,0);
    delete[] row_pointers;delete[] byte_data;

    png_destroy_write_struct(&png_ptr,&info_ptr);
    fclose(file);
}

template<class T> void PngFile<T>::write(const std::string& filename,RawArray<const Vector<T,3>,2> image) {
  write_helper(filename, image);
}

template<class T> void PngFile<T>::write(const std::string& filename,RawArray<const Vector<T,4>,2> image) {
  write_helper(filename, image);
}

template<class T,int C> std::vector<unsigned char> write_to_memory_helper(RawArray<const Vector<T,C>,2> image)
{
  const int width = image.m;
  const int height = image.n;

  typedef std::vector<unsigned char> ByteBuffer;
  ByteBuffer result;
  struct VecWriter {
    static void write(png_structp png_ptr, png_bytep data, png_size_t length) {
      std::vector<unsigned char>* buffer = (ByteBuffer*)png_get_io_ptr(png_ptr);
      buffer->insert(buffer->end(), (unsigned char*)data, (unsigned char*)(data+length));
    }
    static void flush(png_structp png_ptr) { }
  };

  png_structp png_ptr=png_create_write_struct(PNG_LIBPNG_VER_STRING,0,0,0);
  if(!png_ptr) GEODE_FATAL_ERROR("Error writing png file to buffer");

  png_infop info_ptr=png_create_info_struct(png_ptr);
  if(!info_ptr) GEODE_FATAL_ERROR("Error writing png file to buffer");

  if(setjmp(png_jmpbuf(png_ptr))) GEODE_FATAL_ERROR("Error writing png file");

  png_set_write_fn(png_ptr, &result, &VecWriter::write, &VecWriter::flush);

  png_set_IHDR(png_ptr,info_ptr,width,height,8,ColorPolicy<C>::Type,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);

  Vector<unsigned char,C>* byte_data=new Vector<unsigned char,C>[width*height];
  Vector<unsigned char,C>** row_pointers=new Vector<unsigned char,C>*[height];
  for(int j=0;j<height;j++){
    row_pointers[j]=byte_data+width*j;
    for(int i=0;i<width;i++) {
      row_pointers[j][i] = Image<T>::to_byte_color(image(i,height-1-j));
    }
  }
  png_set_rows(png_ptr,info_ptr,(png_byte**)row_pointers);
  png_write_png(png_ptr,info_ptr,PNG_TRANSFORM_IDENTITY,0);
  delete[] row_pointers;
  delete[] byte_data;

  png_destroy_write_struct(&png_ptr,&info_ptr);
  return result;
}

template<class T> std::vector<unsigned char> PngFile<T>::
write_to_memory(RawArray<const Vector<T,3>,2> image) { return write_to_memory_helper(image); }
template<class T> std::vector<unsigned char> PngFile<T>::
write_to_memory(RawArray<const Vector<T,4>,2> image) { return write_to_memory_helper(image); }


//#####################################################################
// Function is_supported
//#####################################################################
template<class T> bool PngFile<T>::
is_supported()
{
    return true;
}
//#####################################################################
#endif
template class PngFile<float>;
template class PngFile<double>;
template class PngFile<unsigned char>;
}
