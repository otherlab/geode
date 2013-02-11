//#####################################################################
// Class Segment<Vector<T,3> >
//##################################################################### 
#include <other/core/geometry/Segment3d.h>
#include <other/core/array/Array.h>
#include <other/core/math/clamp.h>
#include <other/core/structure/Tuple.h>
#include <other/core/vector/Vector.h>
#include <other/core/utility/Log.h>
namespace other{

PyObject* to_python(const Segment<Vector<real,3>>& seg) {
  return to_python(tuple(seg.x0,seg.x1));
}

template<class T> Vector<T,3> Segment<Vector<T,3> >::
closest_point(const Vector<T,3>& point) const
{                  
    Vector<T,3> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return x0; // x0 and x1 are a single point
    else{
        T t=dot(point-x0,v)/denominator;
        if(t <= 0) return x0;
        else if(t >= 1) return x1;
        else{v=x0+(x1-x0)*t;return v;}}
}

template<class T> Vector<T,3> Segment<Vector<T,3> >::
closest_point(const Vector<T,3>& point, Vector<T,2>& weights) const
{                  
    Vector<T,3> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0){
      weights.set(1,0);
      return x0;} // x0 and x1 are a single point
    else{
        T t=clamp(dot(point-x0,v)/denominator,(T)0,(T)1);
        weights.set(1-t,t);
        return (1-t)*x0+t*x1;}
}

template<class T> T Segment<Vector<T,3> >::
distance(const Vector<T,3>& point) const
{                  
    Vector<T,3> v=closest_point(point),d=v-point;
    return d.magnitude();
}

template<class T> Vector<T,3> Segment<Vector<T,3> >::
closest_point_on_line(const Vector<T,3>& point) const
{                  
    Vector<T,3> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return x0; // x0 and x1 are a single point
    else{
        T t=dot(point-x0,v)/denominator;
        v=x0+(x1-x0)*t;return v;}
}

template<class T> T Segment<Vector<T,3> >::
distance_from_point_to_line(const Vector<T,3>& point) const
{                  
    Vector<T,3> v=closest_point_on_line(point),d=v-point;
    return d.magnitude();
}

// vector points from argument segment to class segment; not accurate as the segments become parallel
template<class T> Vector<T,3> Segment<Vector<T,3> >::
shortest_vector_between_lines(const Segment<TV>& segment,Vector<T,2>& weights) const
{
    Vector<T,3> u=x1-x0,v=segment.x1-segment.x0,w=segment.x0-x0;
    T u_sqr_magnitude=u.sqr_magnitude(),v_sqr_magnitude=v.sqr_magnitude(),u_dot_u=u_sqr_magnitude,v_dot_v=v_sqr_magnitude,u_dot_v=dot(u,v),
        u_dot_w=dot(u,w),v_dot_w=dot(v,w);
    T denominator=u_dot_u*v_dot_v-sqr(u_dot_v),rhs1=v_dot_v*u_dot_w-u_dot_v*v_dot_w,rhs2=u_dot_v*u_dot_w-u_dot_u*v_dot_w;
    weights.x=rhs1/denominator;weights.y=rhs2/denominator;
    return weights.x*u-w-weights.y*v;
}

// vector points from argument segment to class segment; not accurate as the segments become parallel
template<class T> Vector<T,3> Segment<Vector<T,3> >::
shortest_vector_between_segments(const Segment<TV>& segment,Vector<T,2>& weights) const
{
    Vector<T,3> u=x1-x0,v=segment.x1-segment.x0,w=segment.x0-x0;
    T u_sqr_magnitude=u.sqr_magnitude(),v_sqr_magnitude=v.sqr_magnitude(),u_dot_u=u_sqr_magnitude,v_dot_v=v_sqr_magnitude,u_dot_v=dot(u,v),
        u_dot_w=dot(u,w),v_dot_w=dot(v,w);
    T denominator=u_dot_u*v_dot_v-sqr(u_dot_v),rhs1=v_dot_v*u_dot_w-u_dot_v*v_dot_w,rhs2=u_dot_v*u_dot_w-u_dot_u*v_dot_w;
    bool check_boundary=false;
    if(rhs1 <= 0 || denominator <= rhs1) check_boundary=true;else weights.x=rhs1/denominator; 
    if(rhs2 <= 0 || denominator <= rhs2) check_boundary=true;else weights.y=rhs2/denominator; 
    if(check_boundary){ // check boundaries of [0,1]x[0,1] weights domain
        T v_plus_w_dot_u=u_dot_v+u_dot_w,u_minus_w_dot_v=u_dot_v-v_dot_w,distance_squared_minus_w_dot_w;
        weights.x=0; // check weights.x=0 side
        if(v_dot_w>=0){distance_squared_minus_w_dot_w=0;weights.y=0;}
        else if(v_dot_v<=-v_dot_w){distance_squared_minus_w_dot_w=v_dot_v+2*v_dot_w;weights.y=1;}
        else{weights.y=-v_dot_w/v_dot_v;distance_squared_minus_w_dot_w=weights.y*v_dot_w;}
        // check weights.x=1 side
        if(u_minus_w_dot_v<=0){T new_distance_squared=u_dot_u-2*u_dot_w;
            if(new_distance_squared<distance_squared_minus_w_dot_w){distance_squared_minus_w_dot_w=new_distance_squared;weights=Vector<T,2>(1,0);}}
        else if(v_dot_v<=u_minus_w_dot_v){T new_distance_squared=v_dot_v+2*(v_dot_w-v_plus_w_dot_u)+u_dot_u;
            if(new_distance_squared<distance_squared_minus_w_dot_w){distance_squared_minus_w_dot_w=new_distance_squared;weights=Vector<T,2>(1,1);}}
        else{T weights_y_temp=u_minus_w_dot_v/v_dot_v,new_distance_squared=u_dot_u-2*u_dot_w-weights_y_temp*u_minus_w_dot_v;
            if(new_distance_squared<distance_squared_minus_w_dot_w){distance_squared_minus_w_dot_w=new_distance_squared;weights=Vector<T,2>(1,weights_y_temp);}}
        // check weights.y=0 side ignoring corners (already handled above)
        if(u_dot_w>0 && u_dot_u>u_dot_w){T weights_x_temp=u_dot_w/u_dot_u,new_distance_squared=-weights_x_temp*u_dot_w;
            if(new_distance_squared<distance_squared_minus_w_dot_w){distance_squared_minus_w_dot_w=new_distance_squared;weights=Vector<T,2>(weights_x_temp,0);}}
        // check weights.y=1 side ignoring corners (already handled above)
        if(v_plus_w_dot_u>0 && u_dot_u>v_plus_w_dot_u){T weights_x_temp=v_plus_w_dot_u/u_dot_u,new_distance_squared=v_dot_v+2*v_dot_w-weights_x_temp*v_plus_w_dot_u;
            if(new_distance_squared<distance_squared_minus_w_dot_w){distance_squared_minus_w_dot_w=new_distance_squared;weights=Vector<T,2>(weights_x_temp,1);}}}
    return weights.x*u-w-weights.y*v;
}

template<class T> T Segment<Vector<T,3> >::
distance(const Segment& segment) const {
  Vector<T,2> weights;
  return magnitude(shortest_vector_between_segments(segment,weights));
}

template<class T> T Segment<Vector<T,3> >::
interpolation_fraction(const Vector<T,3>& location) const
{  
    return Segment::interpolation_fraction(location,x0,x1);
}

template<class T> Vector<T,2> Segment<Vector<T,3> >::
barycentric_coordinates(const Vector<T,3>& location) const
{  
    return Segment::barycentric_coordinates(location,x0,x1);
}

template<class T> Vector<T,2> Segment<Vector<T,3> >::
clamped_barycentric_coordinates(const Vector<T,3>& location,const T tolerance) const
{  
    return Segment::clamped_barycentric_coordinates(location,x0,x1);
}

template<class T> T Segment<Vector<T,3> >::
interpolation_fraction(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1) 
{  
    Vector<T,3> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator==0) return 0; // x0 and x1 are a single point
    else return dot(location-x0,v)/denominator;
}

template<class T> Vector<T,2> Segment<Vector<T,3> >::
barycentric_coordinates(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1)
{  
    T t=interpolation_fraction(location,x0,x1);
    return Vector<T,2>(1-t,t);
}

template<class T> Vector<T,2> Segment<Vector<T,3> >::
clamped_barycentric_coordinates(const Vector<T,3>& location,const Vector<T,3>& x0,const Vector<T,3>& x1,const T tolerance)
{  
    Vector<T,3> v=x1-x0;
    T denominator=dot(v,v);
    if(abs(denominator)<tolerance) return Vector<T,2>((T).5,(T).5);
    T a=clamp(dot(location-x0,v)/denominator,(T)0,(T)1);
    return Vector<T,2>((T)1-a,a);
}

template class Segment<Vector<real,3> >;
}
