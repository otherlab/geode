//#####################################################################
// Class Ray
//#####################################################################
#pragma once

#include <geode/vector/Vector.h>
#include <geode/vector/normalize.h>
namespace geode {

template<class TV> class Ray
{
public:
    typedef typename TV::Scalar T;
    TV start; // start of the ray where t=0
    TV direction; // direction the ray sweeps out - unit vector

    Ray()
     : start(TV())
    {
      direction=TV::axis_vector(TV::dimension-1);
    }

    Ray(const TV& start_input, const TV& direction_input, const bool already_normalized=false)
     : start(start_input)
     , direction(direction_input)
    { if(!already_normalized) direction.normalize(); }

    explicit Ray(const Segment<TV>& segment)
     : start(segment.x0)
     , direction(segment.vector().normalized())
    { }

    TV point(const T t) const // finds the point on the ray, given by the parameter t
    { return start+t*direction; }

    T parameter(const TV& point) const // finds the parameter t, given a point that lies on the ray
    { int axis=direction.dominant_axis();return (point[axis]-start[axis])/direction[axis]; }

    TV reflected_direction(const TV& normal) const
    { return 2*dot(-direction,normal)*normal+direction; }
};

template<class TV> std::ostream &operator<<(std::ostream &output,const Ray<TV> &ray)
{ return output<<"Ray{"<<ray.start<<", "<<ray.direction<<"}"; }

} // geode namespace
