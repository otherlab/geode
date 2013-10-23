//#####################################################################
// Class Sphere
//##################################################################### 
#pragma once

#include <geode/geometry/Box.h>
#include <geode/math/constants.h>
#include <geode/math/pow.h>
#include <geode/python/repr.h>
#include <geode/utility/format.h>
#include <geode/array/Array.h>
namespace geode {

template<class TV>
class Sphere
{
    typedef typename TV::Scalar T;
    enum Workaround {d=TV::m};
public:
    typedef TV VectorT;

    TV center;
    T radius;

    Sphere()
        :radius(1)
    {}

    Sphere(const TV& center,const T radius)
        :center(center),radius(radius)
    {}

    TV normal(const TV& X) const
    {return (X-center).normalized();}

    bool inside(const TV& X,const T thickness_over_two) const
    {return (X-center).sqr_magnitude() <= sqr(radius-thickness_over_two);}

    bool lazy_inside(const TV& X) const
    {return (X-center).sqr_magnitude() <= sqr(radius);}
    
    TV surface(const TV& X) const  
    {return radius*(X-center).normalized()+center;}

    T phi(const TV& X) const
    {return (X-center).magnitude()-radius;}

    T volume() const
    {return (T)unit_sphere_size<d>::value*pow<d>(radius);}

    Box<TV> bounding_box() const
    {return Box<TV>(center).thickened(radius);}

    Vector<T,d-1> principal_curvatures(const TV& X) const
    {return Vector<T,d-1>::all_ones_vector()/radius;}

    bool lazy_intersects(const Box<TV>& box) const
    {return box.phi(center)<=radius;}

    string repr() const
    {return format("Sphere(%s,%s)",tuple_repr(center),geode::repr(radius));}

//#####################################################################
};

template<class TV> Sphere<TV> approximate_bounding_sphere(Array<const TV> X);

}
