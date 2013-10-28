//#####################################################################
// Class Triangle<Vector<T,3> >
//#####################################################################
#pragma once

#include <geode/geometry/Box.h>
#include <geode/geometry/Plane.h>
#include <geode/math/constants.h>
#include <geode/structure/Tuple.h>
namespace geode {

template<class T>
class Triangle<Vector<T,3> >:public Plane<T>
{
    typedef Vector<T,3> TV;
public:
    using Plane<T>::x0;using Plane<T>::n;

    TV x1,x2; // x0 (in Plane), x1 and x2 - clockwise order when looking at the plane

    Triangle()
    {
        n = TV(0,0,1);
        x0 = TV(0,0,0);
        x1 = TV(0,1,0);
        x2 = TV(1,0,0);
    }

    Triangle(const TV& x0_in,const TV& x1_in,const TV& x2_in)
    {
        Plane<T>::operator=(Plane<T>(x0_in,x1_in,x2_in));
        x1=x1_in;x2=x2_in;
    }

    template<class TArray>
    Triangle(const TArray& X)
    {
        BOOST_STATIC_ASSERT(TArray::m==3);
        Plane<T>::operator=(Plane<T>(X[0],X[1],X[2]));
        x1=X[1];x2=X[2];
    }

    T area() const
    {return area(x0,x1,x2);}

    static T area(const TV& x0,const TV& x1,const TV& x2) // always positive for clockwise vertices: x0, x1, x2
    {return (T).5*cross(x1-x0,x2-x0).magnitude();}

    static T sqr_area(const TV& x0,const TV& x1,const TV& x2) // always positive for clockwise vertices: x0, x1, x2
    {return (T).25*cross(x1-x0,x2-x0).sqr_magnitude();}

    T quality() const {
      // compute rms edge length
      real lrms2 = 1/3. * ((x1-x0).sqr_magnitude() + (x2-x0).sqr_magnitude() + (x2-x1).sqr_magnitude());
      real A = area();

      if (A == 0)
        return 0.;
      else {
        // area of a unit equilateral triangle
        const real normalization = 0.25 * sqrt(3.);
        return 1./normalization * A / lrms2;
      }
    }

    T size() const
    {return area();}

    template<class TArray>
    static T size(const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return area(X(0),X(1),X(2));}

    template<class TArray>
    static T signed_size(const TArray& X)
    {return size(X);}

    T aspect_ratio() const
    {return aspect_ratio(x0,x1,x2);}

    static T aspect_ratio(const TV& x1_input,const TV& x2_input,const TV& x3_input)
    {TV u=x1_input-x2_input,v=x2_input-x3_input,w=x3_input-x1_input;
    T u2=dot(u,u),v2=dot(v,v),w2=dot(w,w);
    return max(u2,v2,w2)/cross(u,v).magnitude();}

    TV edge_lengths() const {
      return vec((x2-x1).magnitude(),(x0-x2).magnitude(),(x1-x0).magnitude());
    }

    static T minimum_edge_length(const TV& x0,const TV& x1,const TV& x2)
    {return sqrt(min((x1-x0).sqr_magnitude(),(x2-x0).sqr_magnitude(),(x2-x1).sqr_magnitude()));}

    static T maximum_edge_length(const TV& x0,const TV& x1,const TV& x2)
    {return sqrt(max((x1-x0).sqr_magnitude(),(x2-x0).sqr_magnitude(),(x2-x1).sqr_magnitude()));}

    T minimum_altitude() const
    {return minimum_altitude(x0,x1,x2);}

    static T minimum_altitude(const TV& x0,const TV& x1,const TV& x2)
    {return 2*area(x0,x1,x2)/maximum_edge_length(x0,x1,x2);}

    static TV barycentric_coordinates(const TV& location,const TV& x0,const TV& x1,const TV& x2) // clockwise vertices
    {TV u=x1-x0,v=x2-x0,w=location-x0;
    T u_dot_u=dot(u,u),v_dot_v=dot(v,v),u_dot_v=dot(u,v),
       u_dot_w=dot(u,w),v_dot_w=dot(v,w);
    T denominator=u_dot_u*v_dot_v-sqr(u_dot_v),one_over_denominator;
    if(abs(denominator)>(T)1e-16) one_over_denominator=1/denominator;else one_over_denominator=(T)1e16;
    T a=(v_dot_v*u_dot_w-u_dot_v*v_dot_w)*one_over_denominator,b=(u_dot_u*v_dot_w-u_dot_v*u_dot_w)*one_over_denominator;
    return TV(1-a-b,a,b);}

    static TV clamped_barycentric_coordinates(const TV& location,const TV& x0,const TV& x1,const TV& x2,const T tolerance=1e-7) // clockwise vertices
    {TV u=x1-x0,v=x2-x0,w=location-x0;
    T u_dot_u=dot(u,u),v_dot_v=dot(v,v),u_dot_v=dot(u,v),
       u_dot_w=dot(u,w),v_dot_w=dot(v,w);
    if(abs(u_dot_u)<tolerance){
        if(abs(v_dot_v)<tolerance) return TV(T(1./3),T(1./3),T(1./3)); // single point
        T c=clamp(v_dot_w/v_dot_v,(T)0,(T)1);T a_and_b=(T).5*(1-c);return TV(a_and_b,a_and_b,c);} // x0 and x1 are a single point
    else if(abs(v_dot_v)<tolerance){
        T b=clamp(u_dot_w/u_dot_u,(T)0,(T)1);T a_and_c=(T).5*(1-b);return TV(a_and_c,b,a_and_c);} // x0 and x2 are a single point
    else{
        T denominator=u_dot_u*v_dot_v-sqr(u_dot_v);
        if(abs(denominator)<tolerance){
            if(u_dot_v>0){ // u and v point in the same direction
                if(u_dot_u>u_dot_v){T b=clamp(u_dot_w/u_dot_u,(T)0,(T)1);return TV(1-b,b,0);}
                else{T c=clamp(v_dot_w/v_dot_v,(T)0,(T)1);return TV(1-c,0,c);}}
            else if(u_dot_w>0){T b=clamp(u_dot_w/u_dot_u,(T)0,(T)1);return TV(1-b,b,0);} // u and v point in opposite directions, and w is on the u segment
            else{T c=clamp(v_dot_w/v_dot_v,(T)0,(T)1);return TV(1-c,0,c);}} // u and v point in opposite directions, and w is on the v segment
        T one_over_denominator=1/denominator;
        T a=clamp((v_dot_v*u_dot_w-u_dot_v*v_dot_w)*one_over_denominator,(T)0,(T)1),b=clamp((u_dot_u*v_dot_w-u_dot_v*u_dot_w)*one_over_denominator,(T)0,(T)1);
        return TV(1-a-b,a,b);}}

    template<class TArray>
    static TV barycentric_coordinates(const TV& location,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return barycentric_coordinates(location,X[0],X[1],X[2]);}

    template<class TArray>
    static TV clamped_barycentric_coordinates(const TV& location,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return clamped_barycentric_coordinates(location,X[0],X[1],X[2]);}

    TV sum_barycentric_coordinates(const Triangle& embedded_triangle) const
    {return barycentric_coordinates(embedded_triangle.x0)+barycentric_coordinates(embedded_triangle.x1)+barycentric_coordinates(embedded_triangle.x2);}

    TV barycentric_coordinates(const TV& location) const
    {return barycentric_coordinates(location,x0,x1,x2);}

    TV clamped_barycentric_coordinates(const TV& location) const
    {return clamped_barycentric_coordinates(location,x0,x1,x2);}

    static TV point_from_barycentric_coordinates(const TV& weights,const TV& x0,const TV& x1,const TV& x2) // clockwise vertices
    {return weights.x*x0+weights.y*x1+weights.z*x2;}

    template<class TArray>
    static TV point_from_barycentric_coordinates(const TV& weights,const TArray& X) // clockwise vertices
    {BOOST_STATIC_ASSERT(TArray::m==3);return weights.x*X(0)+weights.y*X(1)+weights.z*X(2);}

    TV point_from_barycentric_coordinates(const TV& weights) const
    {return point_from_barycentric_coordinates(weights,x0,x1,x2);}

    static TV center(const TV& x0,const TV& x1,const TV& x2) // centroid
    {return T(1./3)*(x0+x1+x2);}

    TV center() const // centroid
    {return center(x0,x1,x2);}

    TV incenter() const // intersection of angle bisectors
    {TV el(edge_lengths());T perimeter=el.x+el.y+el.z;assert(perimeter>0);
    return point_from_barycentric_coordinates(el/perimeter);}

    Box<TV> bounding_box() const
    {return geode::bounding_box(x0,x1,x2);}

    const TV& X(const int i) const
    {assert(unsigned(i)<3);
    switch(i){
        case 0:  return x0;
        case 1:  return x1;
        default: return x2;}}

    TV& X(const int i)
    {assert(unsigned(i)<3);
    switch(i){
        case 0:  return x0;
        case 1:  return x1;
        default: return x2;}}

    // For the extremely rare case when one really does need the angles of a triangle
    static TV angles(const TV& x0,const TV& x1,const TV& x2) {
      TV e10 = x0-x1, e12 = x2-x1, e20 = x0-x2;
      T a0 = angle_between(e10,e20),
        a1 = angle_between(e10,e12);
      return vec(a0,a1,pi-a0-a1);
    }

    TV angles() const {
      return angles(x0,x1,x2);
    }

    // For templatization purposes
    static T min_weight(const Vector<T,3>& w) {
      return w.min();
    }

    //#####################################################################
    GEODE_CORE_EXPORT void change_size(const T delta);
    GEODE_CORE_EXPORT bool intersection(Plane<T> const &plane, Segment<Vector<T,3>> &result) const;
    GEODE_CORE_EXPORT bool intersection(Ray<Vector<T,3> >& ray,const T thickness_over_2=0) const;
    GEODE_CORE_EXPORT bool lazy_intersection(Ray<Vector<T,3> >& ray) const;
    GEODE_CORE_EXPORT bool closest_non_intersecting_point(Ray<Vector<T,3> >& ray,const T thickness_over_2=0) const;
    GEODE_CORE_EXPORT bool point_inside_triangle(const TV& point,const T thickness_over_2=0) const;
    GEODE_CORE_EXPORT bool planar_point_inside_triangle(const TV& point,const T thickness_over_2=0) const;
    GEODE_CORE_EXPORT bool lazy_planar_point_inside_triangle(const TV& point) const;
    GEODE_CORE_EXPORT T minimum_edge_length() const;
    GEODE_CORE_EXPORT T maximum_edge_length() const;
    GEODE_CORE_EXPORT Tuple<TV,TV> closest_point(const TV& location) const; // closest_point,weights
    GEODE_CORE_EXPORT T distance(const TV& location) const; // distance from point to triangle
    GEODE_CORE_EXPORT T minimum_angle() const;
    GEODE_CORE_EXPORT T maximum_angle() const;
    GEODE_CORE_EXPORT T signed_solid_angle(const TV& center) const;
    //#####################################################################
};

template<class T> std::ostream& operator<<(std::ostream& output,const Triangle<Vector<T,3> >& triangle)
{output<<'['<<triangle.x0<<','<<triangle.x1<<','<<triangle.x2<<']';return output;}

GEODE_CORE_EXPORT bool intersection(const Segment<Vector<real,3> >& segment, const Triangle<Vector<real,3> >& triangle, const real thickness_over_2=0);

template<class T> static inline Vector<T,3> barycentric_coordinates(const Triangle<Vector<T,3>>& tri, const Vector<T,3>& p) {
  return tri.barycentric_coordinates(p);
}

}
