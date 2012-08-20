//#####################################################################
// Class Plane
//##################################################################### 
#include <other/core/geometry/Plane.h>
#include <other/core/geometry/Ray.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/geometry/Segment3d.h>
namespace other{
//#####################################################################
// Function Intersection
//#####################################################################
// A version to make Box3d::Intersection have to do fewer redundant dot products
template<class T> bool Plane<T>::
intersection(Ray<Vector<T,3> >& ray,const T thickness_over_2,const T distance,const T rate_of_approach) const
{
    if(distance>-thickness_over_2 && distance<thickness_over_2){ray.t_max=0;ray.intersection_location=Ray<Vector<T,3> >::StartPoint;return true;} // within the boundary
    if(rate_of_approach*distance<=0) return false; // no intersection
    if(rate_of_approach>0 && distance-thickness_over_2<ray.t_max*rate_of_approach){
        ray.t_max=(distance-thickness_over_2)/rate_of_approach;ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
    else if(rate_of_approach<0 && distance+thickness_over_2>ray.t_max*rate_of_approach){
        ray.t_max=(distance+thickness_over_2)/rate_of_approach;ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
    return false;
}
//#####################################################################
// Function Intersection
//#####################################################################
template<class T> bool Plane<T>::
intersection(Ray<Vector<T,3> >& ray,const T thickness_over_2) const
{
    T distance=dot(n,ray.start-x0);
    if(distance>-thickness_over_2 && distance<thickness_over_2){ray.t_max=0;ray.intersection_location=Ray<Vector<T,3> >::StartPoint;return true;} // within the boundary
    T rate_of_approach=-dot(n,ray.direction);
    if(rate_of_approach*distance<=0) return false; // no intersection
    if(rate_of_approach>0 && distance-thickness_over_2<ray.t_max*rate_of_approach){
        ray.t_max=(distance-thickness_over_2)/rate_of_approach;ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
    else if(rate_of_approach<0 && distance+thickness_over_2>ray.t_max*rate_of_approach){
        ray.t_max=(distance+thickness_over_2)/rate_of_approach;ray.intersection_location=Ray<Vector<T,3> >::InteriorPoint;return true;}
    return false;
}
  
//#####################################################################
// Function segment_intersection
//#####################################################################
template<class T> typename Plane<T>::TV Plane<T>::
segment_intersection(Segment<TV> const &s) const {
  T interpolation_fraction;
  segment_intersection(s.x0, s.x1, interpolation_fraction);
  return s.interpolated(interpolation_fraction);
}
  
//#####################################################################
// Function Intersection
//#####################################################################
template<class T> template<class TThickness> bool Plane<T>::
intersection(const Box<TV>& box,const TThickness thickness_over_2) const 
{
    bool points_on_positive_side=false,points_on_negative_side=false;
    for(int i=0;i<=1;i++) for(int j=0;j<=1;j++) for(int k=0;k<=1;k++){
        TV test_point(i?box.min.x:box.max.x,j?box.min.y:box.max.y,k?box.min.z:box.max.z);
        T distance=dot(n,test_point-x0);
        if(distance>-thickness_over_2)points_on_positive_side=true;
        if(distance<thickness_over_2)points_on_negative_side=true;}
    return points_on_positive_side && points_on_negative_side;
}
//#####################################################################
// Function Lazy_Intersection
//#####################################################################
template<class T> bool Plane<T>::
lazy_intersection(Ray<Vector<T,3> >& ray) const
{
    T distance=dot(n,ray.start-x0),rate_of_approach=-dot(n,ray.direction);
    if(rate_of_approach*distance<=0) return false; // no intersection
    if(rate_of_approach>0 && distance<ray.t_max*rate_of_approach){ray.t_max=distance/rate_of_approach;return true;}
    else if(rate_of_approach<0 && distance>ray.t_max*rate_of_approach){ray.t_max=distance/rate_of_approach;return true;}
    return false;
}
//#####################################################################
// Segment_Plane_Intersection
//#####################################################################
template<class T> bool Plane<T>::
segment_plane_intersection(const TV& endpoint1,const TV& endpoint2,T& interpolation_fraction) const
{
    T denominator=dot(endpoint2-endpoint1,n);
    if(!denominator){interpolation_fraction=FLT_MAX;return false;} // parallel
    interpolation_fraction=dot(x0-endpoint1,n)/denominator;
    return (interpolation_fraction>=0 && interpolation_fraction<=1);
}
//#####################################################################
// Function Rectangle_Intersection
//#####################################################################
template<class T> bool Plane<T>::
rectangle_intersection(Ray<Vector<T,3> >& ray,const Plane<T>& bounding_plane_1,const Plane<T>& bounding_plane_2,const Plane<T>& bounding_plane_3,const Plane<T>& bounding_plane_4,
                       const T thickness_over_2) const
{
    Ray<Vector<T,3> > ray_temp;ray.save_intersection_information(ray_temp);
    if(intersection(ray,thickness_over_2)){
        TV point=ray.point(ray.t_max);
        if(!bounding_plane_1.outside(point,thickness_over_2) && !bounding_plane_2.outside(point,thickness_over_2) && !bounding_plane_3.outside(point,thickness_over_2)
            && !bounding_plane_4.outside(point,thickness_over_2)) return true;
        else ray.restore_intersection_information(ray_temp);}
    return false;
}
//#####################################################################
template class Plane<real>;
template bool Plane<real>::intersection<Zero>(const Box<Vector<real,3> >&,Zero) const;
template bool Plane<real>::intersection<real>(const Box<Vector<real,3> >&,real) const;
}
