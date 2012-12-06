//#####################################################################
// Class Image
//#####################################################################
#pragma once

#include <other/core/array/Array2d.h>
#include <other/core/vector/Vector3d.h>
#include <vector>
namespace other{

using std::string;
using std::vector;

template<class T> T component_to_scalar_color(const uint8_t color_in) {
  return ((T)color_in + (T).5)/(T)256;
}

template<class T> T component_to_byte_color(const T color_in) {
  return clamp((T)256*color_in,(T)0,(T)255);
}

template<> uint8_t component_to_scalar_color(const uint8_t color_in);
template<> uint8_t component_to_byte_color(const uint8_t color_in);

template<class T, int C> Vector<T,C> to_scalar_color(const Vector<uint8_t, C> color_in) {
  Vector<T,C> result;
  for(int i = 0; i < C; ++i) {
    result[i] = component_to_scalar_color<T>(color_in[i]);
  }
  return result;
}

template<class T, int C> Vector<uint8_t, C> to_byte_color(const Vector<T, C> color_in) {
  Vector<uint8_t,C> result;
  for(int i = 0; i < C; ++i) {
    result[i] = (uint8_t)component_to_byte_color<T>(color_in[i]);
  }
  return result;
}

// Read functions return Arrays constructed from row major data that end up 'transposed' (i.e. width = sizes().y and height = sizes().x) but can be directly passed to openGL
// The write functions currently take column major data and require a transpose() for data comeing from a read or openGL
template<class T>
class Image : public Object
{
public:
    OTHER_DECLARE_TYPE

    template<int C> static Vector<T,C> to_scalar_color(const Vector<uint8_t,C> color_in)
    {return other::to_scalar_color<T,C>(color_in);}
    
    template<int C> static Vector<uint8_t,C> to_byte_color(const Vector<T,C> color_in)
    {return other::to_byte_color<T,C>(color_in); }

    static void flip_x(Array<Vector<T,3>,2>& image)
    {for(int i=0;i<image.m/2;i++)for(int j=0;j<image.n;j++) swap(image(i,j),image(image.m-1-i,j));}

    static void flip_y(Array<Vector<T,3>,2>& image)
    {for(int j=0;j<image.n/2;j++)for(int i=0;i<image.m;i++) swap(image(i,j),image(i,image.n-1-j));}

    static void invert(Array<Vector<T,3>,2>& image)
    {for(int i=0;i<image.m;i++) for(int j=0;j<image.n;j++) image(i,j)=Vector<T,3>::ones()-image(i,j);}

    static void threshold(Array<Vector<T,3>,2>& image,const T threshold,const Vector<T,3>& low_color,const Vector<T,3>& high_color)
    {for(int i=0;i<image.m;i++) for(int j=0;j<image.n;j++) image(i,j)=image(i,j).magnitude()<threshold?low_color:high_color;}

//#####################################################################
    static Array<Vector<T,3>,2> read(const string& filename) OTHER_EXPORT;
    static Array<Vector<T,4>,2> read_alpha(const string& filename) OTHER_EXPORT;
    static void write(const string& filename,RawArray<const Vector<T,3>,2> image) OTHER_EXPORT;
    static void write_alpha(const string& filename,RawArray<const Vector<T,4>,2> image) OTHER_EXPORT;
    static std::vector<unsigned char> write_png_to_memory(RawArray<const Vector<T,3>,2> image) OTHER_EXPORT;
    static std::vector<unsigned char> write_png_to_memory(RawArray<const Vector<T,4>,2> image) OTHER_EXPORT;
    static Array<Vector<T,3>,2> gamma_compress(Array<const Vector<T,3>,2> image,const real gamma) OTHER_EXPORT;
    static Array<Vector<T,3>,2> dither(Array<const Vector<T,3>,2> image) OTHER_EXPORT;
    static bool is_supported(const string& filename) OTHER_EXPORT;
    static Array<Vector<T,3>,2> median(const vector<Array<const Vector<T,3>,2> >& images) OTHER_EXPORT;
//#####################################################################
};
}
