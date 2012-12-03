//#####################################################################
// Class Sphere
//##################################################################### 
#include <other/core/geometry/Sphere.h>
namespace other{

// From Jack ritter: ftp://ftp-graphics.stanford.edu/pub/Graphics/RTNews/html/rtnews7b.html#art4
template<class TV> Sphere<TV> approximate_bounding_sphere(Array<const TV> X)
{
    typedef typename TV::Scalar T;
    if(!X.size()) return Sphere<TV>(TV(),0);
    Box<TV> box = bounding_box(X);
    Sphere<TV> sphere(box.center(),(T).5*box.sizes().max());
    for(int i=0;i<X.size();i++){
        TV DX=X[i]-sphere.center;
        T sqr_distance=DX.sqr_magnitude();
        if(sqr_distance>sqr(sphere.radius)){
            T distance=sqrt(sqr_distance);
            T shift=(T).5*(distance-sphere.radius);
            sphere.center+=shift/distance*DX;
            sphere.radius+=shift;}}
    return sphere;
};

template Sphere<Vector<real,3> > approximate_bounding_sphere(Array<const Vector<real,3> > X);
}
