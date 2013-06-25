//#####################################################################
// Class Triangle<Vector<T,2>>
//##################################################################### 
#include <other/core/geometry/Triangle2d.h>
#include <other/core/geometry/Segment.h>
namespace other{

template<class T> Tuple<Vector<T,2>,Vector<T,3>> Triangle<Vector<T,2>>::
closest_point(const Vector<T,2>& location) const
{
    TV closest;
    Vector<T,3> weights = barycentric_coordinates(location);
    // project closest point to the triangle if it's not already inside it
    if(weights.x<0){
        T a23=interpolation_fraction(simplex(X[1],X[2]),location); // Check edge X[1]--X[2]
        if(a23<0){
            if(weights.z<0){ // Closest point is on edge X[0]--X[1]
                T a12=clamp<T>(interpolation_fraction(simplex(X[0],X[1]),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*X[0]+weights.y*X[1];}
            else{weights=Vector<T,3>(0,1,0);closest=X[1];}} // Closest point is X[1]
        else if(a23>1){
            if(weights.y<0){ // Closest point is on edge X[0]--X[2]
                T a13=clamp<T>(interpolation_fraction(simplex(X[0],X[2]),location),0,1);weights=Vector<T,3>(1-a13,0,a13);closest=weights.x*X[0]+weights.z*X[2];}
            else{weights=Vector<T,3>(0,0,1);closest=X[2];}} // Closest point is X[2]
        else{weights=Vector<T,3>(0,1-a23,a23);closest=weights.y*X[1]+weights.z*X[2];}} // Closest point is on edge X[1]--X[2]
    else if(weights.y<0){
        T a13=interpolation_fraction(simplex(X[0],X[2]),location); // Check edge X[0]--X[2]
        if(a13<0){
            if(weights.z<0){ // Closest point is on edge X[0]--X[1]
                T a12=clamp<T>(interpolation_fraction(simplex(X[0],X[1]),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*X[0]+weights.y*X[1];}
            else{weights=Vector<T,3>(1,0,0);closest=X[0];}} // Closest point is X[0]
        else if(a13>1){weights=Vector<T,3>(0,0,1);closest=X[2];} // Closest point is X[2]
        else{weights=Vector<T,3>(1-a13,0,a13);closest=weights.x*X[0]+weights.z*X[2];}} // Closest point is on edge X[0]--X[2]
    else if(weights.z<0){ // Closest point is on edge X[0]--X[1]
        T a12=clamp<T>(interpolation_fraction(simplex(X[0],X[1]),location),0,1);weights=Vector<T,3>(1-a12,a12,0);closest=weights.x*X[0]+weights.y*X[1];}
    else
        closest=weights.x*X[0]+weights.y*X[1]+weights.z*X[2]; // Point is interior to the triangle
    return tuple(closest,weights);
}

template<class T> T Triangle<Vector<T,2>>::
distance(const TV& location) const {
  return magnitude(location-closest_point(location).x);
}

typedef real T;
template Tuple<Vector<T,2>,Vector<T,3>> Triangle<Vector<T,2>>::closest_point(const Vector<T,2>&) const;
template T Triangle<Vector<real,2>>::distance(const Vector<T,2>&) const;
}
