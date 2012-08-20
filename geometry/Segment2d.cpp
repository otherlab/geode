//#####################################################################
// Class Segment<Vector<T,2> >  
//##################################################################### 
#include <other/core/geometry/Segment2d.h>
#include <other/core/array/Array.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/geometry/Ray.h>
#include <other/core/utility/Log.h>
namespace other{

template<class T> bool Segment<Vector<T,2> >::
segment_line_intersection(const Vector<T,2>& point_on_line,const Vector<T,2>& normal_of_line,T &interpolation_fraction) const
{
    T denominator=dot(x1-x0,normal_of_line);
    if(!denominator){interpolation_fraction=FLT_MAX;return false;} // parallel
    interpolation_fraction=dot(point_on_line-x0,normal_of_line)/denominator;
    return interpolation_fraction<=1 && interpolation_fraction>=0;
}

template<class T> Vector<T,2> Segment<Vector<T,2> >::
closest_point(const Vector<T,2>& point) const
{                  
    Vector<T,2> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return x0; // x0 and x1 are a single point
    else{
        T t=dot(point-x0,v)/denominator;
        if(t <= 0) return x0;
        else if(t >= 1) return x1;
        else{v=x0+(x1-x0)*t;return v;}}
}

template<class T> Vector<T,2> Segment<Vector<T,2> >::
closest_point(const Vector<T,2>& point, Vector<T,2>& weights) const
{                  
    Vector<T,2> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0){
        weights.set(1,0);
        return x0;} // x0 and x1 are a single point
    else{
        T t=clamp(dot(point-x0,v)/denominator,(T)0,(T)1);
        weights.set(1-t,t);
        return (1-t)*x0+t*x1;}
}

template<class T> T Segment<Vector<T,2> >::
distance(const Vector<T,2>& point) const
{                  
    Vector<T,2> v=closest_point(point),d=v-point;
    return d.magnitude();
}

template<class T> Vector<T,2> Segment<Vector<T,2> >::
closest_point_on_line(const Vector<T,2>& point) const
{                  
    Vector<T,2> v=x1-x0;
    T denominator=dot(v,v);
    if(denominator == 0) return x0; // x0 and x1 are a single point
    else{
        T t=dot(point-x0,v)/denominator;
        v=x0+(x1-x0)*t;return v;}
}

template<class T> T Segment<Vector<T,2> >::
distance_from_point_to_line(const Vector<T,2>& point) const
{                  
    Vector<T,2> v=closest_point_on_line(point),d=v-point;
    return d.magnitude();
}

// vector points from input segment to this segment
// not accurate as the segments become parallel
template<class T> Vector<T,2> Segment<Vector<T,2> >::
shortest_vector_between_segments(const Segment<Vector<T,2> >& segment,T& a,T& b) const
{
    Vector<T,2> u=x1-x0,v=segment.x1-segment.x0,w=segment.x0-x0;
    T u_magnitude=u.magnitude(),v_magnitude=v.magnitude();
    T u_dot_u=sqr(u_magnitude),v_dot_v=sqr(v_magnitude),u_dot_v=dot(u,v),
               u_dot_w=dot(u,w),v_dot_w=dot(v,w);
    T denominator=u_dot_u*v_dot_v-sqr(u_dot_v);
    T rhs1=v_dot_v*u_dot_w-u_dot_v*v_dot_w,rhs2=u_dot_v*u_dot_w-u_dot_u*v_dot_w;
    if(rhs1 <= 0) a=0;else if(denominator < rhs1) a=1;else a=rhs1/denominator;
    if(rhs2 <= 0) b=0;else if(denominator < rhs2) b=1;else b=rhs2/denominator;
    if(a > 0 && a < 1){
        if(b == 0){a=u_dot_w/u_dot_u;if(a < 0) a=0;else if(a > 1) a=1;}
        else if(b == 1){a=dot(segment.x1-x0,u)/u_dot_u;if(a < 0) a=0;else if(a > 1) a=1;}}
    else if(b > 0 && b < 1){
        if(a == 0){b=-v_dot_w/v_dot_v;if(b < 0) b=0;else if(b > 1) b=1;}
        else if(a == 1){b=dot(x1-segment.x0,v)/v_dot_v;if(b < 0) b=0;else if(b > 1) b=1;}}
    else{
        T a_distance=(a == 0 ? -rhs1:rhs1-denominator)*u_magnitude,b_distance=(b == 0 ? -rhs2:rhs2-denominator)*v_magnitude;
        if(a_distance > b_distance){
            if(a == 0) b=-v_dot_w/v_dot_v;else b=dot(x1-segment.x0,v)/v_dot_v;if(b < 0) b=0;else if(b > 1) b=1;}
        else{if(b == 0) a=u_dot_w/u_dot_u;else a=dot(segment.x1-x0,u)/u_dot_u;if(a < 0) a=0;else if(a > 1) a=1;}}
    return a*u-w-b*v;
}

template<class T> bool Segment<Vector<T,2> >::
segment_segment_intersection(const Segment<Vector<T,2> >& segment,const T thickness_over_2) const
{
    Ray<Vector<T,2> > ray(segment.x0,segment.x1-segment.x0);ray.t_max=(segment.x1-segment.x0).magnitude();
    return intersection(ray,thickness_over_2);
}

template<class T> int Segment<Vector<T,2> >::
segment_segment_interaction(const Segment<Vector<T,2> >& segment,const Vector<T,2>& v1,const Vector<T,2>& v2,const Vector<T,2>& v3,const Vector<T,2>& v4,const T interaction_distance,
                            T& distance,Vector<T,2>& normal,T& a,T& b,T& relative_speed,const T small_number) const
{
    normal=shortest_vector_between_segments(segment,a,b);
    distance=normal.magnitude();if(distance > interaction_distance) return 0; // no interaction
    Vector<T,2> velocity1=(1-a)*v1+a*v2,velocity2=(1-b)*v3+b*v4;
    if(distance > small_number) normal/=distance;
    else{ // set normal based on relative velocity perpendicular to the two points
        Vector<T,2> relative_velocity=velocity1-velocity2;
        Vector<T,2> u=x1-x0;
        normal=relative_velocity-dot(relative_velocity,u)/dot(u,u)*u;
        T normal_magnitude=normal.magnitude();
        if(normal_magnitude > small_number) normal/=normal_magnitude;
        else{ // relative velocity perpendicular to the segment is 0, pick any direction perpendicular to the segment
            if(abs(u.x) > abs(u.y) && abs(u.x)) normal=Vector<T,2>(0,1);
            else if(abs(u.y) > abs(u.x) && abs(u.y)) normal=Vector<T,2>(1,0);
            else normal=Vector<T,2>(0,1);
            normal=normal-dot(normal,u)/dot(u,u)*u;normal.normalize();
            Log::cout << "                                            Picking Random Normal !!!!!!!!!!!!!!!!!!!!!!!" <<  std::endl;}}
    relative_speed=dot(velocity1-velocity2,normal); // relative speed is in the normal direction
    return 1;
}

// Segment is lengthened at each end by thickness_over_two
template<class T> bool Segment<Vector<T,2> >::
intersection(Ray<Vector<T,2> >& ray,const T thickness_over_two) const 
{
    Vector<T,2> from_start_to_start=x0-ray.start,segment_direction=x1-x0;T segment_length=segment_direction.normalize();
    T cross_product=cross(ray.direction,segment_direction),abs_cross_product=abs(cross_product);
    if(ray.t_max*abs_cross_product>thickness_over_two){
        T cross_recip=((T)1)/cross_product;
        T ray_t=cross_recip*cross(from_start_to_start,segment_direction);
        if (ray_t<0||ray_t>ray.t_max)return false;
        T segment_t=cross_recip*cross(from_start_to_start,ray.direction);
        if (segment_t<-thickness_over_two||segment_t>segment_length+thickness_over_two)return false;
        ray.t_max=ray_t;return true;}
    return false;
}

// Optimized intersection for segment(x0,y),(x1,y), must have x0<x1
// Segment is lengthened at each end by thickness_over_two
template<class T> bool Segment<Vector<T,2> >::
intersection_x_segment(Ray<Vector<T,2> >& ray,const T x0,const T x1,const T y,const T thickness_over_two)
{
    assert(x0<x1);
    Vector<T,2> from_start_to_start(x0-ray.start.x,y-ray.start.y);T segment_length=x1-x0;
    T cross_product=-ray.direction.y,abs_cross_product=abs(cross_product);
    if(ray.t_max*abs_cross_product>thickness_over_two){
        T cross_recip=((T)1)/cross_product;
        T ray_t=-cross_recip*from_start_to_start.y;
        if (ray_t<0||ray_t>ray.t_max)return false;
        T segment_t=cross_recip*cross(from_start_to_start,ray.direction);
        if (segment_t<-thickness_over_two||segment_t>segment_length+thickness_over_two)return false;
        ray.t_max=ray_t;return true;}
    return false;
}

// Optimized intersection for segment (x,y1),(x,y2), must have y1<y2
// Segment is lengthened at each end by thickness_over_two
template<class T> bool Segment<Vector<T,2> >::
intersection_y_segment(Ray<Vector<T,2> >& ray,const T x,const T y1,const T y2,const T thickness_over_two)
{
    assert(y1<y2);
    Vector<T,2> from_start_to_start(x-ray.start.x,y1-ray.start.y);T segment_length=y2-y1;
    T cross_product=ray.direction.x,abs_cross_product=abs(cross_product);
    if(ray.t_max*abs_cross_product>thickness_over_two){
        T cross_recip=((T)1)/cross_product;
        T ray_t=cross_recip*from_start_to_start.x;
        if (ray_t<0||ray_t>ray.t_max)return false;
        T segment_t=cross_recip*cross(from_start_to_start,ray.direction);
        if (segment_t<-thickness_over_two||segment_t>segment_length+thickness_over_two)return false;
        ray.t_max=ray_t;return true;}
    return false;
}

template<class T> bool Segment<Vector<T,2> >::
linear_point_inside_segment(const TV& X,const T thickness_over_2) const
{
    Vector<T,2> weights=barycentric_coordinates(X);
    return weights.x>=-thickness_over_2 && weights.y>=-thickness_over_2;
}

// outputs unsigned distance
template<class T> bool Segment<Vector<T,2> >::
point_face_interaction(const Vector<T,2>& x,const T interaction_distance,T& distance) const
{      
    distance=dot(x-x0,normal());
    return abs(distance)<=interaction_distance && linear_point_inside_segment(x,interaction_distance);
}

template<class T> void Segment<Vector<T,2> >::
point_face_interaction_data(const Vector<T,2>& x,T& distance,Vector<T,2>& interaction_normal,Vector<T,2>& weights,const bool perform_attractions) const
{
    interaction_normal=normal();weights=barycentric_coordinates(x);
    if(!perform_attractions && distance<0){distance*=-1;interaction_normal*=-1;} // distance > 0, interaction_normal points from the triangle to the point
}

template<class T> bool Segment<Vector<T,2> >::
point_face_interaction(const Vector<T,2>& x,const Vector<T,2>& v,const TV& v1,const TV& v2,const T interaction_distance,T& distance,
    Vector<T,2>& interaction_normal,Vector<T,2>& weights,T& relative_speed,const bool exit_early) const
{
    if(!point_face_interaction(x,interaction_distance,distance)) return false;
    if(!exit_early){
        point_face_interaction_data(x,distance,interaction_normal,weights,false);
        relative_speed=dot(v-(weights.x*v1+weights.y*v2),interaction_normal);} // relative speed is in the normal direction
    return true;
}

template class Segment<Vector<real,2> >;
}
