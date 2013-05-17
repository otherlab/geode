//#####################################################################
// Class Segment<Vector<T,2> >
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/geometry/forward.h>
#include <other/core/geometry/Box.h>
#include <other/core/vector/Vector2d.h>
#include <other/core/vector/normalize.h>
namespace other {

OTHER_EXPORT PyObject* to_python(const Segment<Vector<real,2>>& seg);

template<class T> inline Vector<T,2> normal(const Vector<T,2>& x0,const Vector<T,2>& x1) {
  return rotate_right_90(normalized(x1-x0));
}

template<class TArray> inline typename EnableForSize<2,TArray,typename TArray::Element>::type normal(const TArray& X) {
  return normal(X[0],X[1]);
}

template<class T>
class Segment<Vector<T,2> >
{
    typedef Vector<T,2> TV;
public:
    TV x0,x1;

    Segment()
    {}

    Segment(const TV& x0,const TV& x1)
        :x0(x0),x1(x1)
    {}

    template<class TArray> explicit Segment(const TArray& X)
        :x0(X[0]),x1(X[1])
    {
      BOOST_STATIC_ASSERT(TArray::m==2);
    }

    T length() const
    {return (x1-x0).magnitude();}

    T size() const
    {return length();}

    template<class TArray>
    static T size(const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==2);return (X(1)-X(0)).magnitude();}

    template<class TArray>
    static T signed_size(const TArray& X)
    {return size(X);}

    TV vector() const
    {return x1-x0;}

    TV interpolated(real t) const
    {return x0 + vector() * t;}

    TV center() const
    {return (T).5*(x0+x1);}

    static TV normal(const TV& x0,const TV& x1)
    {return rotate_right_90(normalized(x1-x0));}

    TV normal() const
    {return Segment::normal(x0,x1);}

    template<class TArray>
    static TV normal(const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==2);return normal(X(0),X(1));}

    static T interpolation_fraction(const TV& location,const TV& x0,const TV& x1)
    {TV v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return 0; // x0 and x1 are a single point
    else return dot(location-x0,v)/denominator;}

    static TV barycentric_coordinates(const TV& location,const TV& x0,const TV& x1)
    {T t=interpolation_fraction(location,x0,x1);
    return TV(1-t,t);}

    static TV clamped_barycentric_coordinates(const TV& location,const TV& x0,const TV& x1)
    {TV v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return TV(1,0); // x0 and x1 are a single point
    else{
        T t=clamp(dot(location-x0,v)/denominator,(T)0,(T)1);
        return TV(1-t,t);}}

    template<class TArray>
    static TV clamped_barycentric_coordinates(const TV& location,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==2);return clamped_barycentric_coordinates(location,X(0),X(1));}

    TV sum_barycentric_coordinates(const Segment& embedded_segment) const
    {return barycentric_coordinates(embedded_segment.x0)+barycentric_coordinates(embedded_segment.x1);}

    TV barycentric_coordinates(const TV& location) const
    {return barycentric_coordinates(location,x0,x1);}

    static TV point_from_barycentric_coordinates(const T alpha,const TV& x0,const TV& x1)
    {return (x1-x0)*alpha+x0;}

    template<class TArray>
    static TV point_from_barycentric_coordinates(const TV& weights,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==2);return weights.x*X(0)+weights.y*X(1);}

    TV point_from_barycentric_coordinates(const T alpha) const
    {return (x1-x0)*alpha+x0;}

    template<class TArray>
    static TV point_from_barycentric_coordinates(const T alpha,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==2);return point_from_barycentric_coordinates(alpha,X(0),X(1));}

    bool point_face_interaction(const TV& x,const TV& v,const IndirectArray<RawArray<TV>,Vector<int,2>&> V_face,const T interaction_distance,T& distance,
            TV& interaction_normal,TV& weights,T& relative_speed,const bool exit_early) const
    {return point_face_interaction(x,v,V_face[1],V_face[2],interaction_distance,distance,interaction_normal,weights,relative_speed,exit_early);}

    Box<TV> bounding_box() const
    {return other::bounding_box(x0,x1);}

    const TV& X(const int i) const
    {assert(unsigned(i)<2);return (&x0)[i];}

    TV& X(const int i)
    {assert(unsigned(i)<2);return (&x0)[i];}

    bool segment_line_intersection(const TV& point_on_line,const TV& normal_of_line,T &interpolation_fraction) const;
    TV closest_point(const TV& point) const;
    TV closest_point(const TV& point, Vector<T,2>& weights) const;
    T distance(const TV& point) const; // distance from point to segment
    T distance(const Segment& segment) const; // distance between segments

    TV closest_point_on_line(const TV& point) const;
    T distance_from_point_to_line(const TV& point) const;
    TV shortest_vector_between_segments(const Segment<TV>& segment,T& a,T& b) const;
    OTHER_CORE_EXPORT bool segment_segment_intersection(const Segment<TV>& segment,const T thickness_over_2=0) const;
    int segment_segment_interaction(const Segment<TV>& segment,const TV& v1,const TV& v2,const TV& v3,const TV& v4,
        const T interaction_distance,T& distance,TV& normal,T& a,T& b,T& relative_speed,const T small_number=0) const;
    bool intersection(Ray<Vector<T,2> >& ray,const T thickness_over_two) const;
    static bool intersection_x_segment(Ray<Vector<T,2> >& ray,const T x0,const T x1,const T y,const T thickness_over_two);
    static bool intersection_y_segment(Ray<Vector<T,2> >& ray,const T x,const T y1,const T y2,const T thickness_over_two);
    bool linear_point_inside_segment(const TV& X,const T thickness_over_2) const;
    bool point_face_interaction(const TV& x,const T interaction_distance,T& distance) const;
    void point_face_interaction_data(const TV& x,T& distance,TV& interaction_normal,TV& weights,const bool perform_attractions) const;
    bool point_face_interaction(const TV& x,const TV& v,const TV& v1,const TV& v2,const T interaction_distance,T& distance,
                                TV& interaction_normal,TV& weights,T& relative_speed,const bool exit_early) const;
};

template<class T> std::ostream &operator<<(std::ostream &output,const Segment<Vector<T,2> > &segment)
{output << '[' << segment.x0 << ',' << segment.x1 << ']';return output;}

}
