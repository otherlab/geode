//#####################################################################
// Class Triangle<Vector<T,3> >
//##################################################################### 
#include <geode/geometry/Triangle3d.h>
#include <geode/array/Array.h>
#include <geode/math/constants.h>
#include <geode/geometry/Ray.h>
#include <geode/geometry/Segment.h>
#include <geode/math/lerp.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;

// Enlarges the triangle by pushing out the triangle edges by distance 'delta' orthogonally to the edges.
// This keeps the incenter fixed.  If the triangle is degenerate, it will not be changed.
template<class T> void Triangle<Vector<T,3> >::
change_size(const T delta)
{
    Vector<T,3> edge_lengths((x2-x1).magnitude(),(x0-x2).magnitude(),(x1-x0).magnitude());
    T perimeter=edge_lengths.sum(),area_=area();
    if(!perimeter || !area_) return; // don't know which direction to enlarge a degenerate triangle, so do nothing
    T scale=1+delta*(T).5*perimeter/area_;
    Vector<T,3> incenter=point_from_barycentric_coordinates(edge_lengths/perimeter);
    x0=incenter+(x0-incenter)*scale;x1=incenter+(x1-incenter)*scale;x2=incenter+(x2-incenter)*scale;
}

template<class T> bool Triangle<Vector<T,3>>::
intersection(Plane<T> const &plane, Segment<Vector<T,3>> &result) const {
  double d0 = plane.phi(x0);
  double d1 = plane.phi(x1);
  double d2 = plane.phi(x2);
  int sd0 = (int)sign(d0);
  int sd1 = (int)sign(d1);
  int sd2 = (int)sign(d2);

  if (sd0 == sd1 && sd1 == sd2)
    return false;

  if (sd0 == sd1) {
    result.x0 = lerp(0, d2, d0, x2, x0);
    result.x1 = lerp(0, d2, d1, x2, x1);
  } else if (sd1 == sd2) {
    result.x0 = lerp(0, d0, d1, x0, x1);
    result.x1 = lerp(0, d0, d2, x0, x2);
  } else {
    result.x0 = lerp(0, d1, d0, x1, x0);
    result.x1 = lerp(0, d1, d2, x1, x2);
  }

  return true;
}

template<class T> bool Triangle<Vector<T,3> >::
intersection(Ray<Vector<T,3> >& ray,const T thickness_over_2) const
{
    Ray<Vector<T,3> > ray_temp;ray.save_intersection_information(ray_temp);
    T thickness=2*thickness_over_2;

    // first check the plane of the triangle's face
    if(!Plane<T>::intersection(ray,thickness_over_2)) return false; // otherwise intersects the plane
    Vector<T,3> plane_point(ray.point(ray.t_max));
    Plane<T> edge_plane_12(cross(x1-x0,n).normalized(),x0),edge_plane_23(cross(x2-x1,n).normalized(),x1),
                     edge_plane_31(cross(x0-x2,n).normalized(),x2);
    if(!edge_plane_12.outside(plane_point,thickness) && !edge_plane_23.outside(plane_point,thickness) && !edge_plane_31.outside(plane_point,thickness))return true; // intersects face of triangle wedge 
    else ray.restore_intersection_information(ray_temp);

    // check for intersection with the sides of the wedge
    if(edge_plane_12.outside(plane_point,thickness_over_2) && edge_plane_12.intersection(ray,thickness_over_2)){
        Vector<T,3> edge_point(ray.point(ray.t_max));
        if(Plane<T>::boundary(edge_point,thickness) && !edge_plane_23.outside(edge_point,thickness) && !edge_plane_31.outside(edge_point,thickness)){
            ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
        else ray.restore_intersection_information(ray_temp);}
    if(edge_plane_23.outside(plane_point,thickness_over_2) && edge_plane_23.intersection(ray,thickness_over_2)){
        Vector<T,3> edge_point(ray.point(ray.t_max));
        if(Plane<T>::boundary(edge_point,thickness) && !edge_plane_12.outside(edge_point,thickness) && !edge_plane_31.outside(edge_point,thickness)){
            ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
        else ray.restore_intersection_information(ray_temp);}
    if(edge_plane_31.outside(plane_point,thickness_over_2) && edge_plane_31.intersection(ray,thickness_over_2)){
        Vector<T,3> edge_point(ray.point(ray.t_max));
        if(Plane<T>::boundary(edge_point,thickness) && !edge_plane_12.outside(edge_point,thickness) && !edge_plane_23.outside(edge_point,thickness)){
            ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
        else ray.restore_intersection_information(ray_temp);}

    return false;
}

template<class T> bool Triangle<Vector<T,3> >::
lazy_intersection(Ray<Vector<T,3> >& ray) const
{
    T save_t_max=ray.t_max;int save_aggregate_id=ray.aggregate_id;
    if(!Plane<T>::lazy_intersection(ray)) return false; // otherwise intersects the plane
    if(lazy_planar_point_inside_triangle(ray.point(ray.t_max))) return true; // intersects the face of the triangle 
    else{ray.t_max=save_t_max;ray.aggregate_id=save_aggregate_id;return false;} // reset ray
}

template<class T> bool Triangle<Vector<T,3> >::
closest_non_intersecting_point(Ray<Vector<T,3> >& ray,const T thickness_over_2) const 
{
    Ray<Vector<T,3> > ray_temp;ray.save_intersection_information(ray_temp);
    if(!intersection(ray,thickness_over_2)) return false;
    else if(ray.intersection_location==Ray<Vector<T,3> >::StartPoint) return true;
    else ray.restore_intersection_information(ray_temp);

    // Todo: Save having to re-generate all the planes...
    T thickness=2*thickness_over_2;
    Vector<T,3> normal_times_thickness=n*thickness;
    Triangle<Vector<T,3> > top_triangle(x0+normal_times_thickness,x1+normal_times_thickness,x2+normal_times_thickness);
    Triangle<Vector<T,3> > bottom_triangle(x0-normal_times_thickness,x1-normal_times_thickness,x2-normal_times_thickness);
    Plane<T> edge_plane_12(cross(x1-x0,n).normalized(),x0);edge_plane_12.x0+=edge_plane_12.n*thickness;
    Plane<T> edge_plane_23(cross(x2-x1,n).normalized(),x1);edge_plane_23.x0+=edge_plane_23.n*thickness;
    Plane<T> edge_plane_31(cross(x0-x2,n).normalized(),x2);edge_plane_31.x0+=edge_plane_31.n*thickness;
    bool found_intersection=false;
    if(top_triangle.intersection(ray,thickness_over_2)) found_intersection=true;
    if(bottom_triangle.intersection(ray,thickness_over_2)) found_intersection=true;
    if(edge_plane_12.rectangle_intersection(ray,top_triangle,bottom_triangle,edge_plane_23,edge_plane_31,thickness_over_2)) found_intersection=true;
    if(edge_plane_23.rectangle_intersection(ray,top_triangle,bottom_triangle,edge_plane_12,edge_plane_31,thickness_over_2)) found_intersection=true;
    if(edge_plane_31.rectangle_intersection(ray,top_triangle,bottom_triangle,edge_plane_12,edge_plane_23,thickness_over_2)) found_intersection=true;
    return found_intersection;
}

template<class T> bool Triangle<Vector<T,3> >::
point_inside_triangle(const Vector<T,3>& point,const T thickness_over_2) const
{
    return Plane<T>::boundary(point,thickness_over_2)&&planar_point_inside_triangle(point,thickness_over_2);
}

template<class T> bool Triangle<Vector<T,3> >::
planar_point_inside_triangle(const Vector<T,3>& point,const T thickness_over_2) const
{
    Plane<T> edge_plane(cross(x1-x0,n).normalized(),x0);if(edge_plane.outside(point,thickness_over_2)) return false;
    edge_plane.n=cross(x0-x2,n).normalized();if(edge_plane.outside(point,thickness_over_2)) return false;
    edge_plane.n=cross(x2-x1,n).normalized();edge_plane.x0=x1;if(edge_plane.outside(point,thickness_over_2)) return false;
    return true;
}

template<class T> bool Triangle<Vector<T,3> >::
lazy_planar_point_inside_triangle(const Vector<T,3>& point) const
{
    Vector<T,3> edge_normal_1=cross(x1-x0,n),point_minus_x1=point-x0;if(dot(edge_normal_1,point_minus_x1) > 0) return false;
    Vector<T,3> edge_normal_2=cross(x0-x2,n);if(dot(edge_normal_2,point_minus_x1) > 0) return false;
    edge_normal_1+=edge_normal_2;if(dot(edge_normal_1,point-x1) < 0) return false; // this equals x1-x2 (== -edge_normal_3)
    return true;
}

template<class T> T Triangle<Vector<T,3> >::
minimum_edge_length() const
{      
    return min((x1-x0).magnitude(),(x2-x1).magnitude(),(x0-x2).magnitude());
}

template<class T> T Triangle<Vector<T,3> >::
maximum_edge_length() const
{      
    return max((x1-x0).magnitude(),(x2-x1).magnitude(),(x0-x2).magnitude());
}

template<class T> Tuple<Vector<T,3>,Vector<T,3>> Triangle<Vector<T,3> >::
closest_point(const Vector<T,3>& location) const
{
    TV closest;
    TV weights=barycentric_coordinates(location);
    // project closest point to the triangle if it's not already inside it
    if(weights.x<0){
        T a23=interpolation_fraction(simplex(x1,x2),location); // Check edge x1--x2
        if(a23<0){
            if(weights.z<0){ // Closest point is on edge x0--x1
                T a12=clamp<T>(interpolation_fraction(simplex(x0,x1),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*x0+weights.y*x1;}
            else{weights=Vector<T,3>(0,1,0);closest=x1;}} // Closest point is x1
        else if(a23>1){
            if(weights.y<0){ // Closest point is on edge x0--x2
                T a13=clamp<T>(interpolation_fraction(simplex(x0,x2),location),0,1);weights=Vector<T,3>(1-a13,0,a13);closest=weights.x*x0+weights.z*x2;}
            else{weights=Vector<T,3>(0,0,1);closest=x2;}} // Closest point is x2
        else{weights=Vector<T,3>(0,1-a23,a23);closest=weights.y*x1+weights.z*x2;}} // Closest point is on edge x1--x2
    else if(weights.y<0){
        T a13=interpolation_fraction(simplex(x0,x2),location); // Check edge x0--x2
        if(a13<0){
            if(weights.z<0){ // Closest point is on edge x0--x1
                T a12=clamp<T>(interpolation_fraction(simplex(x0,x1),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*x0+weights.y*x1;}
            else{weights=Vector<T,3>(1,0,0);closest=x0;}} // Closest point is x0
        else if(a13>1){weights=Vector<T,3>(0,0,1);closest=x2;} // Closest point is x2
        else{weights=Vector<T,3>(1-a13,0,a13);closest=weights.x*x0+weights.z*x2;}} // Closest point is on edge x0--x2
    else if(weights.z<0){ // Closest point is on edge x0--x1
        T a12=clamp<T>(interpolation_fraction(simplex(x0,x1),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*x0+weights.y*x1;}
    else
        closest=weights.x*x0+weights.y*x1+weights.z*x2; // Point is interior to the triangle
    return tuple(closest,weights);
}

template<class T> T Triangle<Vector<T,3>>::distance(const Vector<T,3>& location) const {   
    return magnitude(location-closest_point(location).x);
}

template<class T> T Triangle<Vector<T,3> >::
minimum_angle() const
{
    Vector<T,3> s1=(x0-x1).normalized(),s2=(x1-x2).normalized(),s3=(x2-x0).normalized();
    return acos(max(dot(s1,-s2),dot(-s1,s3),dot(s2,-s3)));
}

template<class T> T Triangle<Vector<T,3> >::
maximum_angle() const
{
    Vector<T,3> s1=(x0-x1).normalized(),s2=(x1-x2).normalized(),s3=(x2-x0).normalized();
    return acos(min(dot(s1,-s2),dot(-s1,s3),dot(s2,-s3)));
}

// positive for normals that point away from the center - not reliable if center is too close to the triangle face
template<class T> T Triangle<Vector<T,3> >::
signed_solid_angle(const Vector<T,3>& center) const
{
    Vector<T,3> r=(x0-center).normalized(),u=x1-x0,v=x2-x0;u-=dot(u,r)*r;v-=dot(v,r)*r;
    T solid_angle=-(T)pi+angle_between(u,v);
    r=(x1-center).normalized();u=x0-x1,v=x2-x1;u-=dot(u,r)*r;v-=dot(v,r)*r;
    solid_angle+=angle_between(u,v);
    r=(x2-center).normalized();u=x0-x2,v=x1-x2;u-=dot(u,r)*r;v-=dot(v,r)*r;
    solid_angle+=angle_between(u,v);
    solid_angle=max(T(0),min((T)(2*pi),solid_angle));
    if(dot(r,n) < 0) solid_angle*=(-1);
    return solid_angle;
}

bool intersection(const Segment<TV>& segment, const Triangle<TV>& triangle, const T thickness_over_2) {
  Ray<TV> ray(segment);
  return triangle.intersection(ray,thickness_over_2);
}

template class Triangle<Vector<real,3> >;
}
