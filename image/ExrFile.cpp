#include <other/core/python/config.h> // Must be included first
#ifdef USE_OPENEXR
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#endif

#include <other/core/image/ExrFile.h>
#include <other/core/image/Image.h>
#include <other/core/array/Array2d.h>
#include <other/core/vector/Vector3d.h>
#include <other/core/utility/Log.h>

namespace other {
#ifndef USE_OPENEXR

  template<class T> Array<Vector<T,3>,2> ExrFile<T>::
  read(const std::string& filename)
  {
    OTHER_FATAL_ERROR("Not compiled with USE_OPENEXR.  Cannot read exr image.");
  }
  
  template<class T> void ExrFile<T>::
  write(const std::string& filename,RawArray<const Vector<T,3>,2> image)
  {
    OTHER_FATAL_ERROR("Not compiled with USE_OPENEXR.  Cannot write exr image.");
  }
  
  template<class T> bool ExrFile<T>::
  is_supported()
  {return false;}
  
#else

  using namespace Imf;

  template<class T>
  Imf::Rgba to_rgba(Vector<T,3> const &v) {
    Imf::Rgba rgba;
    rgba.r = (float)v.x;
    rgba.g = (float)v.y;
    rgba.b = (float)v.z;
    rgba.a = 1;
    return rgba;
  }

  template<>
  Imf::Rgba to_rgba(Vector<unsigned char,3> const &v) {
    Imf::Rgba rgba;
    rgba.r = v.x/255;
    rgba.g = v.y/255;
    rgba.b = v.z/255;
    rgba.a = 1;
    return rgba;
  }

  template<class T>
  Vector<T,3> from_rgba(Imf::Rgba const &rgba) {
    return Vector<T,3>(rgba.r,rgba.g,rgba.b);
  }
  
  template<>
  Vector<unsigned char,3> from_rgba(Imf::Rgba const &rgba) {
    return Vector<unsigned char,3>(vec(255*rgba.r,255*rgba.g,255*rgba.b));
  }
  
  template<class T> Array<Vector<T,3>,2> ExrFile<T>::
  read(const std::string& filename)
  {
    Imf::RgbaInputFile file(filename.c_str());
    Imath::Box2i dw = file.dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    Imf::Array2D<Imf::Rgba> pixels;
    pixels.resizeErase(height, width);
    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
    
    Array<Vector<T,3>,2> image(width, height);
    for (int i = 0; i < image.m; ++i) {
      for (int j = 0; j < image.n; ++j) {
        image(i,j) = from_rgba<T>(pixels[j][i]);
      }
    }
    return image;
  }
  
  template<class T> void ExrFile<T>::
  write(const std::string& filename, RawArray<const Vector<T,3>,2> image)
  {
    // convert to array of EXRPixels
    Imf::Rgba *pixels = new Imf::Rgba[image.total_size()];
    
    for (int i = 0; i < image.m; ++i) {
      for (int j = 0; j < image.n; ++j) {
        pixels[j*image.m+i] = to_rgba(image(i,j));
      }
    }
    
    Imf::RgbaOutputFile file(filename.c_str(), image.m, image.n, Imf::WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, image.m);
    file.writePixels(image.n);
    
    delete[] pixels;
  }
  
  template<class T> bool ExrFile<T>::
  is_supported()
  {
    return true;
  }  
  
#endif
    
  template class ExrFile<float>;
  template class ExrFile<double>;
  template class ExrFile<unsigned char>;
}
