//#####################################################################
// Class Segment<Vector<T,3> >
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/geometry/Box.h>
#include <other/core/vector/Vector3d.h>
namespace other {

template<class T>
class Segment<Vector<T,3>> {
  typedef Vector<T,3> TV;
public:
  Vector<T,3> x0,x1;

  Segment()
  {}

  Segment(const Vector<T,3>& x0,const Vector<T,3>& x1)
      :x0(x0),x1(x1)
  {}

  template<class TArray>
  Segment(const TArray& X)
      :x0(X[0]),x1(X[1])
  {
      BOOST_STATIC_ASSERT(TArray::m==2);
  }

  T length() const
  {return (x1-x0).magnitude();}

  T size() const
  {return length();}

  static T size(const TV& x0,const TV& x1)
  {return (x1-x0).magnitude();}

  template<class TArray>
  static T signed_size(const TArray& X)
  {BOOST_STATIC_ASSERT(TArray::m==2);return size(X(0),X(1));}

  template<class TArray>
  static T size(const TArray& X)
  {return signed_size(X);}

  TV vector() const
  {return x1-x0;}

  TV interpolated(real t) const
  {return x0 + vector() * t;}

  Box<TV> bounding_box() const
  {return other::bounding_box(x0,x1);}

  OTHER_CORE_EXPORT Vector<T,3> closest_point(const Vector<T,3>& point) const;
  OTHER_CORE_EXPORT Vector<T,3> closest_point(const Vector<T,3>& point,Vector<T,2>& weights) const;
  OTHER_CORE_EXPORT T distance(const Vector<T,3>& point) const; // distance from point to segment
  OTHER_CORE_EXPORT T distance(const Segment& segment) const; // distance between segments
  OTHER_CORE_EXPORT Vector<T,3> closest_point_on_line(const Vector<T,3>& point) const;
  OTHER_CORE_EXPORT T distance_from_point_to_line(const Vector<T,3>& point) const;
  OTHER_CORE_EXPORT Vector<T,3> shortest_vector_between_lines(const Segment& segment,Vector<T,2>& weights) const;
  OTHER_CORE_EXPORT Vector<T,3> shortest_vector_between_segments(const Segment& segment,Vector<T,2>& weights) const;
  OTHER_CORE_EXPORT T interpolation_fraction(const Vector<T,3>& location) const;
  OTHER_CORE_EXPORT Vector<T,2> barycentric_coordinates(const Vector<T,3>& location) const;
  OTHER_CORE_EXPORT Vector<T,2> clamped_barycentric_coordinates(const Vector<T,3>& location,const T tolerance=1e-7) const;
  OTHER_CORE_EXPORT static T interpolation_fraction(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1);
  OTHER_CORE_EXPORT static Vector<T,2> barycentric_coordinates(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1);
  OTHER_CORE_EXPORT static Vector<T,2> clamped_barycentric_coordinates(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1,const T tolerance=1e-7);
};

template<class TV> std::ostream& operator<<(std::ostream& output, Segment<TV> const &s)
{return output<<'['<<s.x0<<','<<s.x1<<']';}

}
