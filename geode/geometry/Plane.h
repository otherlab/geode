//#####################################################################
// Class Plane
//#####################################################################
#pragma once

#include <geode/geometry/forward.h>
#include <geode/vector/Vector3d.h>
#include <geode/math/Zero.h>
namespace geode {

template<class T> inline Vector<T,3> normal(const Vector<T,3>& x0,const Vector<T,3>& x1,const Vector<T,3>& x2)
{return cross(x1-x0,x2-x0).normalized();}

template<class TArray> inline typename EnableForSize<3,TArray,typename TArray::Element>::type normal(const TArray& X)
{return normal(X[0],X[1],X[2]);}

template<class T>
class Plane
{
    typedef Vector<T,3> TV;
public:
    typedef TV VectorT;

    TV n;
    TV x0; // point on the plane

    Plane()
        :n(0,1,0)
    {}

    Plane(const TV& normal,const TV& x0)
        :n(normal),x0(x0)
    {}

    Plane(const TV& x0_input,const TV& x1_input,const TV& x2_input)
    {
        n=geode::normal(x0_input,x1_input,x2_input);x0=x0_input;
    }

    TV normal() const
    {return n;}

    TV normal(const TV& X) const
    {return n;}

    static TV normal_direction(const TV& x0,const TV& x1,const TV& x2)
    {return cross(x1-x0,x2-x0);} // can have any magnitude

    template<class TArray>
    static TV normal_direction(const TArray& X)
    {StaticAssert(TArray::m==3);return normal_direction(X(0),X(1),X(2));}

    T phi(const TV& location) const
    {return dot(n,location-x0);}

    // inside is the half space behind the normal
    bool inside(const TV& location,const T thickness_over_two) const
    {return phi(location)<=-thickness_over_two;}

    bool lazy_inside(const TV& location) const
    {return phi(location)<=0;}

    bool outside(const TV& location,const T thickness_over_two) const
    {return !inside(location,-thickness_over_two);}

    bool lazy_outside(const TV& location) const
    {return !lazy_inside(location);}

    bool boundary(const TV& location,const T thickness_over_two) const
    {return abs(phi(location))<thickness_over_two;}

    // closest point on the surface
    TV surface(const TV& location) const
    {return location-phi(location)*n;}

    TV mirror(const TV& location) const
    {return location-2*phi(location)*n;}

    bool segment_intersection(const TV& endpoint1,const TV& endpoint2,T& interpolation_fraction) const
    {return segment_plane_intersection(endpoint1,endpoint2,interpolation_fraction);}

    GEODE_CORE_EXPORT TV segment_intersection(Segment<TV> const &s) const;

    Box<TV> bounding_box() const
    {return Box<TV>::full_box();}

    Vector<T,2> principal_curvatures(const TV& X) const
    {return Vector<T,2>();}

    static std::string name()
    {return "Plane<T>";}

    bool intersection(Plane<T> const &, RayIntersection<Vector<T,3> > &ray) const;
    bool intersection(RayIntersection<Vector<T,3> >& ray,const T thickness_over_2,const T distance,const T rate_of_approach) const;
    bool intersection(RayIntersection<Vector<T,3> >& ray,const T thickness_over_2=0) const;
    template<class TThickness> bool intersection(const Box<TV>& box,const TThickness thickness_over_2=Zero()) const;
    bool segment_plane_intersection(const TV& endpoint1,const TV& endpoint2,T& interpolation_fraction) const;
    bool lazy_intersection(RayIntersection<Vector<T,3> >& ray) const;
    bool rectangle_intersection(RayIntersection<Vector<T,3> >& ray,const Plane<T>& bounding_plane_1,const Plane<T>& bounding_plane_2,const Plane<T>& bounding_plane_3,const Plane<T>& bounding_plane_4,
        const T thickness_over_2=0) const;

    string repr() const
    {return format("Plane(%s,%s)",tuple_repr(n),tuple_repr(x0));}
};

template<class T> std::ostream & operator<<(std::ostream & os, Plane<T> const &p) {
  return os << '[' << p.x0 << ", " << p.n << ']' << std::endl;
}

}
