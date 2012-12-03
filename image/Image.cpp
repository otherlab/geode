//#####################################################################
// Class Image
//#####################################################################
#include <other/core/geometry/Box.h>
#include <other/core/image/Image.h>
#include <other/core/image/JpgFile.h>
#include <other/core/image/PngFile.h>
#include <other/core/image/ExrFile.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/random/Random.h>
#include <other/core/utility/path.h>
#include <other/core/utility/Log.h>
#include <cmath>

namespace other {

using std::pow;
using std::nth_element;

template<> OTHER_DEFINE_TYPE(Image<float>)
template<> OTHER_DEFINE_TYPE(Image<double>)

template<class T> Array<Vector<T,3>,2> Image<T>::read(const string& filename)
{
  string ext = path::extension(filename);
  if(ext==".jpg" || ext==".jpeg") return JpgFile<T>::read(filename);
  else if(ext==".png") return PngFile<T>::read(filename);
  else if(ext==".exr") return ExrFile<T>::read(filename);
  OTHER_FATAL_ERROR(format("Unknown image file extension  from filename '%s' extension '%s'",filename,ext));
}

template<class T> Array<Vector<T,4>,2> Image<T>::read_alpha(const string& filename)
{
  string ext = path::extension(filename);
  if(ext==".png") return PngFile<T>::read_alpha(filename);
  OTHER_FATAL_ERROR(format("Image file extension unknown or without alpha from filename '%s' extension '%s'",filename,ext));
}

template<class T> void Image<T>::
write(const string& filename,RawArray<const Vector<T,3>,2> image)
{
    string ext = path::extension(filename);
    if(ext==".jpg" || ext==".jpeg") JpgFile<T>::write(filename,image);
    else if(ext==".png") PngFile<T>::write(filename,image);
    else if(ext==".exr") ExrFile<T>::write(filename,image);
    else OTHER_FATAL_ERROR(format("Unknown image file extension from filename '%s' extension '%s'",filename,ext));
}

template<class T> void Image<T>::
write_alpha(const string& filename,RawArray<const Vector<T,4>,2> image)
{
    string ext = path::extension(filename);
    if(ext==".png") PngFile<T>::write(filename,image);
    else if(ext==".jpg") OTHER_FATAL_ERROR(format("No support for alpha channel with extension '%s' for filename '%s'",ext,filename));
    else OTHER_FATAL_ERROR(format("Unknown image file extension from filename '%s' extension '%s'",filename,ext));
}

template<class T> std::vector<unsigned char> Image<T>::
write_png_to_memory(RawArray<const Vector<T,3>,2> image) {
  return PngFile<T>::write_to_memory(image);
}

template<class T> std::vector<unsigned char> Image<T>::
write_png_to_memory(RawArray<const Vector<T,4>,2> image) {
  return PngFile<T>::write_to_memory(image);
}

template<class T> Array<Vector<T,3>,2> Image<T>::
gamma_compress(Array<const Vector<T,3>,2> image,const real gamma)
{
    T one_over_gamma = T(1/gamma);
    Array<Vector<T,3>,2> result(image.sizes());
    for(int t=0;t<result.flat.size();t++)
      for (int i = 0; i < 3; ++i)
        result.flat[t][i] = pow(image.flat(t)[i],one_over_gamma);
    return result;
}

template<class T> Array<Vector<T,3>,2> Image<T>::
dither(Array<const Vector<T,3>,2> image)
{
    Ref<Random> random=new_<Random>(324032); // want uniform seed so noise perturbation pattern is temporally coherent

    Array<Vector<T,3>,2> result(image.sizes());
    for(int t=0;t<result.flat.size();t++){
        result.flat(t)=image.flat(t);
        Vector<T,3> pixel_values((T)255*result.flat(t));
        Vector<int,3> floored_values((int)pixel_values[0],(int)pixel_values[1],(int)pixel_values[2]);
        Vector<real,3> random_stuff=random->uniform<Vector<real,3> >(0,1);
        Vector<T,3> normalized_values=pixel_values-Vector<T,3>(floored_values);
        for(int k=0;k<3;k++)
            if(random_stuff[k]>normalized_values[k]) result.flat(t)[k]=(floored_values[k]+(T).5001)/255; // use normal quantized floor
            else result.flat(t)[k]=(floored_values[k]+(T)1.5001)/255;} // jump to next value
    return result;
}

template<class T>
Array<Vector<T,3>,2> Image<T>::median(const vector<Array<const Vector<T,3>,2> >& images) {
  OTHER_ASSERT(images.size());
  int n = (int)images.size();
  for (int k=1;k<n;k++)
    OTHER_ASSERT(images[0].sizes()==images[k].sizes());

  Array<T,2> pixel(3,n);
  Array<Vector<T,3>,2> result(images[0].sizes());
  for(int t=0; t<result.flat.size(); t++){
    for (int k=0;k<n;k++)
      images[k].flat[t].get(pixel(0,k),pixel(1,k),pixel(2,k));
    for (int a=0;a<3;a++) {
      RawArray<T> samples = pixel[a];
      nth_element(&samples[0],&samples[n/2],&samples[n-1]);
      result.flat[t][a] = samples[n/2];
    }
  }
  return result;
}

template<class T> bool Image<T>::
is_supported(const string& filename)
{
    string ext = path::extension(filename);
    if(ext==".jpg" || ext==".jpeg") return JpgFile<T>::is_supported();
    else if(ext==".png") return PngFile<T>::is_supported();
    else if(ext==".exr") return ExrFile<T>::is_supported();
    else return false;
}

template<> uint8_t component_to_scalar_color(const uint8_t color_in) { return color_in; }
template<> uint8_t component_to_byte_color(const uint8_t color_in) { return color_in; }

template class Image<float>;
template class Image<double>;
template Array<Vector<uint8_t,3>,2> Image<uint8_t>::read(const string&);
template Array<Vector<uint8_t,4>,2> Image<uint8_t>::read_alpha(const string&);
template void Image<uint8_t>::write(const string&,RawArray<const Vector<uint8_t,3>,2>);
template void Image<uint8_t>::write_alpha(const string&,RawArray<const Vector<uint8_t,4>,2>);

template vector<unsigned char> Image<uint8_t>::write_png_to_memory(RawArray<const Vector<uint8_t,3>,2>);
template vector<unsigned char> Image<uint8_t>::write_png_to_memory(RawArray<const Vector<uint8_t,4>,2>);

}


void wrap_Image()
{
  using namespace other;
  using namespace python;
  typedef real T;

  typedef Image<T> Self;
  Class<Self>("Image")
    .OTHER_METHOD(read)
    .OTHER_METHOD(write)
    .OTHER_METHOD(gamma_compress)
    .OTHER_METHOD(dither)
    .OTHER_METHOD(median)
    .OTHER_METHOD(is_supported)
    ;
}
