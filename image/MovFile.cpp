//#####################################################################
// Class MovFile
//#####################################################################
#ifdef _WIN32
// we need config.h to not undefine far, which is used in jpeglib.h
#define LEAVE_WINDOWS_DEFINES_ALONE
#endif
#include <other/core/python/config.h>
#ifdef USE_LIBJPEG
extern "C"{
#ifdef _WIN32
#undef HAVE_STDDEF_H
#endif
#include <jpeglib.h>
}
#undef HAVE_PROTOTYPES
#undef HAVE_STDLIB_H
#endif
#include <other/core/python/Class.h>
#include <other/core/image/Image.h>
#include <other/core/image/MovFile.h>
#include <boost/detail/endian.hpp>
#include <string>
#include <iostream>
#include <cassert>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
namespace other{

OTHER_DEFINE_TYPE(MovWriter)

typedef unsigned int uint;
typedef unsigned short ushort;

template<class T>
static inline void convert_endian(T& x)
{
    BOOST_STATIC_ASSERT(sizeof(T)<=8);
#ifndef BOOST_BIG_ENDIAN
    char* p = (char*)&x;
    for(uint i=0;i<sizeof(T)/2;i++)
        swap(p[i],p[sizeof(T)-1-i]);
#endif
}

template<> inline void convert_endian(char& x){}

template<class T>
static void write(FILE* fp, T num)
{
    convert_endian(num);
    fwrite(&num,sizeof(num),1,fp);
}

static void write_identity_matrix(FILE* fp)
{
    write(fp,(uint)0x10000);write(fp,(uint)0x00000);write(fp,(uint)0); // 16.16 fixed pt
    write(fp,(uint)0x00000);write(fp,(uint)0x10000);write(fp,(uint)0); // 16.16 fixed pt
    write(fp,(uint)0x00000);write(fp,(uint)0x00000);write(fp,(uint)0x40000000); // 2.30 fixed pt
}

class QtAtom
{
    FILE *fp;
    long start_offset;
    //const char* type;
public:
    QtAtom(FILE* fp,const char* type)
        :fp(fp)//,type(type)
    {
        start_offset=ftell(fp);
        uint dummy;
        fwrite(&dummy,4,1,fp);
        fputs(type,fp);
    }

    ~QtAtom()
    {
        uint atom_size=uint(ftell(fp)-start_offset);
        convert_endian(atom_size);
        fseek(fp,start_offset,SEEK_SET);
        fwrite(&atom_size,4,1,fp);
        fseek(fp,0,SEEK_END);
    }

    long offset()
    {return start_offset;}
};

MovWriter::
MovWriter(const std::string& filename,const int frames_per_second)
    :frames_per_second(frames_per_second),width(0),height(0)
{
    OTHER_ASSERT(enabled());
    fp=fopen(filename.c_str(),"wb");
    if(!fp) OTHER_FATAL_ERROR(format("Failed to open %s for writing",filename));
    current_mov=new QtAtom(fp,"mdat");
}

MovWriter::
~MovWriter()
{
    delete current_mov;
    write_footer();
    fclose(fp);
}

void MovWriter::
add_frame(const Array<Vector<T,3>,2>& image)
{
#ifdef USE_LIBJPEG
/*
    static int frame=0;
    Image<T>::write(format("capture.%02d.jpg",frame),image);
    frame++;
*/

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    if(width==0 && height==0){width=image.m;height=image.n;}
    if(width!=image.m || height!=image.n) throw RuntimeError("Frame does not have same size as previous frame(s)");

    cinfo.err=jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    long frame_begin=ftell(fp);
    jpeg_stdio_dest(&cinfo,fp);
    cinfo.image_width=image.m;
    cinfo.image_height=image.n;
    cinfo.input_components=3;
    cinfo.in_color_space=JCS_RGB; // colorspace of input image
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo,95,TRUE); // limit to baseline-Jpeg values
    jpeg_start_compress(&cinfo,TRUE);

    int row_stride=cinfo.image_width*3; // JSAMPLEs per row in image_buffer
    JSAMPLE* row=new JSAMPLE[row_stride];
    JSAMPROW row_pointer[]={row};
    while(cinfo.next_scanline < cinfo.image_height){
        int index=0;
        for(int i=0;i<image.m;i++){ // copy row
            Vector<unsigned char,3> pixel=Image<T>::to_byte_color(image(i,image.n-cinfo.next_scanline-1));
            row[index++]=pixel.x;row[index++]=pixel.y;row[index++]=pixel.z;}
        jpeg_write_scanlines(&cinfo,row_pointer,1);}
    delete[] row;
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    long frame_end=ftell(fp);
    sample_lengths.append(int(frame_end-frame_begin));
    sample_offsets.append(int(frame_begin-current_mov->offset()));
#endif
}

void MovWriter::
write_footer()
{
    const int frames=sample_offsets.size();
    OTHER_ASSERT(sample_offsets.size()==sample_lengths.size());
    QtAtom a(fp,"moov");
        {QtAtom a(fp,"mvhd");
            char c=0;
            write(fp,c); // version
            write(fp,c);write(fp,c);write(fp,c); // reserved
            write(fp,(uint)0); // creation time
            write(fp,(uint)0); // modification time
            write(fp,(uint)frames_per_second); // time rate 1/30th second
            write(fp,(uint)frames); // duration
            write(fp,(uint)0x10000); // preferred rate 16bit fixed pt 1.0
            write(fp,(ushort)0x100); // full volume
            write(fp,(uint)0);write(fp,(uint)0);write(fp,(ushort)0x0); // 10 bytes padded
            write_identity_matrix(fp);
            write(fp,(uint)0); // preview time
            write(fp,(uint)0); // preview duration
            //write(fp,(uint)0); // poster time - for some reason, commenting out this line fixes the code
            write(fp,(uint)0); // selection time
            write(fp,(uint)frames); // selection duration
            write(fp,(uint)0); // current time
            write(fp,(uint)2);} // next track
        {QtAtom a(fp,"trak");
            {QtAtom a(fp,"tkhd");
                write(fp,(uint)0xf); // flag visibble
                write(fp,(uint)0); // creation time
                write(fp,(uint)0); // modification time
                write(fp,(uint)1); // track id
                write(fp,(uint)0); // reserved
                write(fp,(uint)frames); // duration
                write(fp,(uint)0);write(fp,(uint)0); // reserved
                write(fp,(ushort)0); // layer
                write(fp,(ushort)0); // alternative group
                write(fp,(ushort)0x100); // volume
                write(fp,(ushort)0); // reserved
                write_identity_matrix(fp);
                write(fp,(uint)width<<16); // width
                write(fp,(uint)height<<16);} // height
            {QtAtom a(fp,"edts");
              {QtAtom a(fp,"elst");
                write(fp,(uint)0); // version flags
                write(fp,(uint)0);}} // 1 entry
            {QtAtom a(fp,"mdia");
                {QtAtom a(fp,"mdhd");
                    write(fp,(uint)0x0); // version/flag visibble
                    write(fp,(uint)0); // creation time
                    write(fp,(uint)0); // modified time
                    write(fp,(uint)frames_per_second); // time scale
                    write(fp,(uint)frames); // duration
                    write(fp,(ushort)0); // english language
                    write(fp,(ushort)0xffff);} // quality
                {QtAtom a(fp,"hdlr");
                    write(fp,(uint)0x0); // version/flags
                    fputs("mhlrvide",fp);
                    write(fp,(uint)0); // component manufacture
                    write(fp,(uint)0); // component flags
                    write(fp,(uint)0); // component flags mask
                    write(fp,(char)0); // component name
                    fputs("Linux Video Media handler",fp);}
                {QtAtom a(fp,"minf");
                    {QtAtom a(fp,"vmhd");
                        write(fp,(uint)0x0001); // version/flags set 1 for compatibility
                        write(fp,(ushort)0x40); // graphics mode copy
                        write(fp,(ushort)0x8000); // unused graphics mode opcolor
                        write(fp,(ushort)0x8000); // unused graphics mode opcolor
                        write(fp,(ushort)0x8000);} // unused graphics mode opcolor
                    {QtAtom a(fp,"hdlr");
                        write(fp,(uint)0x0); // version/flags
                        fputs("dhlralis",fp);
                        write(fp,(uint)0); // component manufacture
                        write(fp,(uint)0); // component flags
                        write(fp,(uint)0); // component flags mask
                        write(fp,(char)0); // component name
                        fputs("Linux Alias Data handler",fp);}
                    {QtAtom a(fp,"dinf");
                        {QtAtom a(fp,"dref");
                            write(fp,(uint)0x0); // vvvf version flags
                            write(fp,(uint)0x1); // 1 entry
                            {QtAtom a(fp,"alis");
                                write(fp,(uint)1);}}}
                    {QtAtom a(fp,"stbl");
                        {QtAtom a(fp,"stsd");
                            write(fp,(uint)0); // version and flags
                            write(fp,(uint)1); // 1 entry
                            {QtAtom a(fp,"jpeg");
                                write(fp,(uint)0); //reserved
                                write(fp,(ushort)0); //reserved
                                write(fp,(ushort)1); // data reference index
                                // write video specific data
                                write(fp,(ushort)0); // version
                                write(fp,(ushort)0); // revision level
                                fputs("lnux",fp); // vendor
                                write(fp,(uint)100); //temporal quality (max)
                                write(fp,(uint)258); //spatial quality (max)
                                write(fp,(ushort)width); // width of image
                                write(fp,(ushort)height); // height of image
                                write(fp,(uint)0x00480000); // height of image (72dpi)
                                write(fp,(uint)0x00480000); // height of image (72dpi)
                                write(fp,(uint)0); // data size (must be zero)
                                write(fp,(ushort)1); // frames per sample (usually 1)
                                const char* descript="Quicktime for linux";
                                write(fp,(char)strlen(descript));
                                OTHER_ASSERT(strlen(descript)<32);
                                fputs(descript,fp); // compressor
                                for(size_t i=0;i<32-strlen(descript)-1;i++) write(fp,(char)0);
                                write(fp,(ushort)24); // color depth
                                write(fp,(ushort)(short)-1);}} // use default color table id
                        {QtAtom a(fp,"stts");
                            write(fp,(uint)0); // version and flags
                            write(fp,(uint)1); // 1 entry
                            write(fp,(uint)frames); // all frames have same duration
                            write(fp,(uint)1);} // duration is one time unit
                        {QtAtom a(fp,"stsc");
                            write(fp,(uint)0); // version and flags
                            write(fp,(uint)1); // 1 entry
                            write(fp,(uint)1); // first sample to use in chunk
                            write(fp,(uint)1); // number of samples per chunk
                            write(fp,(uint)1);} // index of descriptor (points to stsd)
                        {QtAtom a(fp,"stsz");
                            write(fp,(uint)0); // version and flags
                            write(fp,(uint)0); // sample size (non-uniform so zero and table follows)
                            write(fp,(uint)frames); // one entry per frame
                            for(int i=0;i<sample_lengths.size();i++) write(fp,(uint)sample_lengths(i));}
                        {QtAtom a(fp,"stco");
                            write(fp,(uint)0); // version and flags
                            write(fp,(uint)frames); // one entry per frame
                            for(int i=0;i<sample_offsets.size();i++) write(fp,(uint)sample_offsets(i));}}}}} // offset from begin of file
}

bool MovWriter::
enabled()
{
#ifdef USE_LIBJPEG
    return true;
#else
    return false;
#endif
}
}

using namespace other;

void wrap_mov()
{
    typedef MovWriter Self;
    Class<Self>("MovWriter")
        .OTHER_INIT(const string&,int)
        .OTHER_METHOD(add_frame)
        .OTHER_METHOD(write_footer)
        .OTHER_METHOD(enabled)
        ;
}
