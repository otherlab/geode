//#####################################################################
// Class JpgFile
//#####################################################################

#ifdef _WIN32
#define LEAVE_WINDOWS_DEFINES_ALONE
#endif

#include <other/core/image/JpgFile.h>
#include <other/core/array/Array2d.h>
#include <other/core/vector/Vector3d.h>

#ifdef USE_LIBJPEG

#include <stdio.h>
extern "C"{
#ifdef _WIN32
#undef HAVE_STDDEF_H
#endif
#include <jpeglib.h>
}
#undef HAVE_PROTOTYPES
#undef HAVE_STDLIB_H
#include <other/core/image/Image.h>
#include <other/core/utility/Log.h>
//#####################################################################
// Function Read
//#####################################################################

namespace other {

static void read_error(j_common_ptr cinfo)
{
    throw IOError("JpgFile:: Can't read image");
}

template<class T> Array<Vector<T,3>,2> JpgFile<T>::
read(const std::string& filename)
{
  //note: this operates like a matrix, not an image; i.e. indexing in a la M by N matrix, not X by Y image
    struct jpeg_decompress_struct cinfo;FILE * infile;int row_stride;struct jpeg_error_mgr error_manager;
    if(!(infile=fopen(filename.c_str(),"rb"))) throw IOError(format("JpgFile::read: Can't open %s",filename));
    cinfo.err=jpeg_std_error(&error_manager);error_manager.error_exit=read_error;
    jpeg_create_decompress(&cinfo);jpeg_stdio_src(&cinfo,infile);jpeg_read_header(&cinfo,TRUE);jpeg_start_decompress(&cinfo);

    row_stride=cinfo.output_width*cinfo.output_components;
    JSAMPLE* row=new unsigned char[row_stride];JSAMPROW row_pointer[]={row};
    Log::cerr<<"reading "<<filename<<": "<<row_stride/3<<" x "<<cinfo.output_height<<std::endl;

    Array<Vector<T,3>,2> image(cinfo.output_height,cinfo.output_width,false);
    while(cinfo.output_scanline<cinfo.output_height){
        jpeg_read_scanlines(&cinfo,row_pointer,1);int index=0;
        for(int i=0;i<image.n;i++){
            unsigned char r=row[index++],g=row[index++],b=row[index++];
            image(image.m-cinfo.output_scanline,i)=Image<T>::to_scalar_color(Vector<unsigned char,3>(r,g,b));}}
    jpeg_finish_decompress(&cinfo);jpeg_destroy_decompress(&cinfo);delete[] row;

    fclose(infile);
    return image;
}
//#####################################################################
// Function Write
//#####################################################################
template<class T> void JpgFile<T>::
write(const std::string& filename,RawArray<const Vector<T,3>,2> image)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile; // target file

    cinfo.err=jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    if(!(outfile = fopen(filename.c_str(), "wb")))
        OTHER_FATAL_ERROR(format("JpgFile::write: Can't open %s",filename));
    jpeg_stdio_dest(&cinfo,outfile);
    cinfo.image_width=image.m;cinfo.image_height=image.n;cinfo.input_components=3;
    cinfo.in_color_space=JCS_RGB; // colorspace of input image
    jpeg_set_defaults(&cinfo);jpeg_set_quality(&cinfo,95,TRUE); // limit to baseline-Jpeg values
    jpeg_start_compress(&cinfo,TRUE);

    int row_stride=cinfo.image_width*3; // JSAMPLEs per row in image_buffer
    JSAMPLE* row=new unsigned char[row_stride];JSAMPROW row_pointer[]={row};
    while(cinfo.next_scanline < cinfo.image_height){
        int index=0;for(int i=0;i<image.m;i++){Vector<unsigned char,3> pixel=Image<T>::to_byte_color(image(i,image.n-1-cinfo.next_scanline));row[index++]=pixel.x;row[index++]=pixel.y;row[index++]=pixel.z;} // copy row
        jpeg_write_scanlines(&cinfo,row_pointer,1);}
    delete[] row;
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}
//#####################################################################
// Function is_supported
//#####################################################################
template<class T> bool JpgFile<T>::
is_supported()
{
    return true;
}

#else

namespace other {

//#####################################################################
// Function Read
//#####################################################################
template<class T> Array<Vector<T,3>,2> JpgFile<T>::
read(const std::string& filename)
{
    OTHER_FATAL_ERROR("Not compiled with USE_LIBJPEG.  Cannot read jpeg image.");
}
//#####################################################################
// Function Write
//#####################################################################
template<class T> void JpgFile<T>::
write(const std::string& filename,RawArray<const Vector<T,3>,2> image)
{
    OTHER_FATAL_ERROR("Not compiled with USE_LIBJPEG.  Cannot write jpeg image.");
}
//#####################################################################
// Function is_supported
//#####################################################################
template<class T> bool JpgFile<T>::
is_supported()
{
    return false;
}
//#####################################################################
#endif

template class JpgFile<float>;
template class JpgFile<double>;
template class JpgFile<unsigned char>;

}
